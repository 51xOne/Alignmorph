"""Endpoint reference memories for scheduled self-attention."""

import math

import torch

import torch.nn.functional as F

from diffusers.models.attention_processor import AttnProcessor2_0

from .endpoint_attention import EndpointInterpolatedAttention

from .inversion import invert_ddim_step


class ReferenceCollector:
    """Store unwarped endpoint latents for each frame."""

    def __init__(self, source, target, frames):
        self.source, self.target = source, target
        self.frames = frames
        self.operands = {}
        self.weights = {}

    def capture(self, index, first, second, reliability):
        self.operands[index] = torch.cat((first, second)).detach().clone()
        if 0 < index < self.frames - 1:
            self.weights[index] = reliability.detach().clone()

    def latents(self):
        if set(self.operands) != set(range(self.frames)):
            raise ValueError("Missing endpoint reference latents")
        return torch.cat(
            [
                self.operands[i][branch : branch + 1]
                for branch in range(2)
                for i in range(self.frames)
            ]
        )

    def reliability(self):
        if set(self.weights) != set(range(1, self.frames - 1)):
            raise ValueError("Missing interior-frame reliability maps")
        zero = torch.zeros_like(self.weights[1])
        return torch.stack(
            [zero] + [self.weights[i] for i in range(1, self.frames - 1)] + [zero]
        )


def attention_modules(unet):
    return [
        (name, module)
        for name, module in unet.named_modules()
        if name.endswith(("attn1", "attn2"))
    ]


def tokens(attn, hidden, temb):
    if attn.spatial_norm is not None:
        hidden = attn.spatial_norm(hidden, temb)
    if hidden.ndim == 4:
        hidden = hidden.flatten(2).transpose(1, 2)
    if attn.group_norm is not None:
        hidden = attn.group_norm(hidden.transpose(1, 2)).transpose(1, 2)
    return hidden


class CaptureProcessor:
    def __init__(self, memory, name):
        self.memory, self.name = memory, name
        self.original = AttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        hidden = tokens(attn, hidden_states, temb)
        self.memory[self.name] = (attn.to_k(hidden), attn.to_v(hidden))
        return self.original(
            attn, hidden_states, attention_mask=attention_mask, temb=temb
        )


class EndpointMemoryProcessor:
    def __init__(self, memory, name, coefficients, is_fused=False):
        self.memory, self.name = memory, name
        self.coefficients, self.is_fused = coefficients, is_fused

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError(
                "Endpoint memories are used only in unmasked self-attention"
            )
        residual, shape = hidden_states, hidden_states.shape
        hidden = tokens(attn, hidden_states, temb)
        frames = hidden.shape[0]
        key, value = self.memory[self.name]
        query = attn.to_q(hidden)
        own_key, own_value = attn.to_k(hidden), attn.to_v(hidden)
        source_key, target_key = key.chunk(2)
        source_value, target_value = value.chunk(2)

        # The first and last lanes retain their own endpoint memory.
        source_key = torch.cat((own_key[:1], source_key[1:]))
        source_value = torch.cat((own_value[:1], source_value[1:]))
        target_key = torch.cat((target_key[:-1], own_key[-1:]))
        target_value = torch.cat((target_value[:-1], own_value[-1:]))
        if self.is_fused:
            source_key, target_key = [
                torch.cat((own_key, k), dim=1) for k in (source_key, target_key)
            ]
            source_value, target_value = [
                torch.cat((own_value, v), dim=1) for v in (source_value, target_value)
            ]

        def heads(tensor):
            return tensor.view(
                frames, -1, attn.heads, tensor.shape[-1] // attn.heads
            ).transpose(1, 2)

        query = heads(query)

        def attend(k, v):
            output = F.scaled_dot_product_attention(
                query,
                heads(k),
                heads(v),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            return (
                output.transpose(1, 2)
                .reshape(frames, -1, attn.heads * query.shape[-1])
                .to(query.dtype)
            )

        coef = self.coefficients.to(query).reshape(frames, 1, 1)
        output = (1 - coef) * attend(source_key, source_value) + coef * attend(
            target_key, target_value
        )
        output = attn.to_out[1](attn.to_out[0](output))
        if len(shape) == 4:
            output = output.transpose(1, 2).reshape(shape)
        if attn.residual_connection:
            output = output + residual
        return output / attn.rescale_output_factor


@torch.no_grad()
def invert_references(unet, clean, text, scheduler, timesteps):
    """Build conditional reference states at the actual denoising timesteps."""
    modules = attention_modules(unet)
    original = [(module, module.processor) for _, module in modules]
    states = {}
    try:
        for _, module in modules:
            module.set_processor(AttnProcessor2_0())
        current = clean
        for t in timesteps:
            current = invert_ddim_step(
                current,
                lambda x, step: unet(x, step, encoder_hidden_states=text).sample,
                scheduler,
                t,
            )
            states[int(t)] = current.detach().clone()
    finally:
        for module, processor in original:
            module.set_processor(processor)
    return states


class ReferenceBank:
    def __init__(self, states, text, reliability):
        self.states, self.text, self.reliability = states, text, reliability
        self.memory = {}

    @torch.no_grad()
    def capture(self, unet, timestep):
        modules = attention_modules(unet)
        original = [(module, module.processor) for _, module in modules]
        self.memory.clear()
        try:
            for name, module in modules:
                module.set_processor(
                    CaptureProcessor(self.memory, name)
                    if name.endswith("attn1")
                    else AttnProcessor2_0()
                )
            unet(self.states[int(timestep)], timestep, encoder_hidden_states=self.text)
        finally:
            for module, processor in original:
                module.set_processor(processor)
