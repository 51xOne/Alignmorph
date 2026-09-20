"""Confidence-gated endpoint memory mixing."""

import math
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from .attention import EndpointMemoryProcessor, attention_modules, ReferenceCollector
from .endpoint_attention import EndpointInterpolatedAttention
from .sampling import generate_beta_tensor
from .geometry import sample_scalar
from .transport import eq8_transport

MAX_ALIGNED_WEIGHT = 0.25


def smooth_gate(x):
    u = ((x - 0.1) / 0.1).clamp(0, 1)
    return u.square() * (3 - 2 * u)


def confidence_gates(corr, frames=7):
    ws, wt = corr.source_reliability, corr.target_reliability
    cs = torch.minimum(
        ws, sample_scalar(wt, corr.source_to_target, padding_mode="zeros")
    ).clamp(0, 1)
    ct = torch.minimum(
        wt, sample_scalar(ws, corr.target_to_source, padding_mode="zeros")
    ).clamp(0, 1)
    gs, gt = smooth_gate(cs), smooth_gate(ct)
    # Same source-chart midpoint convention as the paper-phase reference branch.
    gates = torch.stack(
        [
            (
                torch.zeros_like(gs)
                if i in (0, frames - 1)
                else gs if i <= frames // 2 else gt
            )
            for i in range(frames)
        ]
    )
    return gates, dict(
        source_confidence=cs, target_confidence=ct, source_gate=gs, target_gate=gt
    )


class ConfidenceProcessor:
    def __init__(
        self,
        original_memory,
        aligned_memory,
        name,
        coefficients,
        gates,
        beta,
        is_fused=False,
        cache=None,
    ):
        if not 0 <= beta <= 1:
            raise ValueError("beta must be in [0,1]")
        self.original = EndpointMemoryProcessor(
            original_memory, name, coefficients, is_fused
        )
        self.aligned = EndpointMemoryProcessor(
            aligned_memory, name, coefficients, is_fused
        )
        self.gates, self.beta = gates, beta
        self.cache = cache if cache is not None else {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        original = self.original(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )
        if self.beta == 0:
            return original
        if hidden_states.ndim == 4:
            h, w = hidden_states.shape[-2:]
        else:
            n = hidden_states.shape[1]
            h = w = math.isqrt(n)
            if h * w != n:
                raise ValueError("Expected square spatial token grid")
        key = (h, w, hidden_states.ndim, original.dtype, original.device)
        if key not in self.cache:
            gate = F.interpolate(
                self.gates[:, None].float(), size=(h, w), mode="area"
            ).to(original)
            if hidden_states.ndim == 3:
                gate = gate.flatten(2).transpose(1, 2)
            self.cache[key] = gate
        gate = self.cache[key]
        aligned = self.aligned(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )
        # Output projection is linear and inference dropout is zero. Shared residuals cancel.
        return original + (self.beta * gate) * (aligned - original)


@torch.no_grad()
def denoise_mixed(
    pipe, latent, text, empty_text, timesteps, original_bank, aligned_bank, gates, beta
):
    coefficients = generate_beta_tensor(pipe.frames, alpha=20.0, beta=20.0)
    first, second = int(len(timesteps) * 0.2), int(len(timesteps) * 0.6)
    cache = {}
    for index, timestep in enumerate(timesteps):
        active = index < second
        if active:
            original_bank.capture(pipe.unet, timestep)
            aligned_bank.capture(pipe.unet, timestep)
            fused = index >= first
            cross = EndpointInterpolatedAttention(is_fused=fused, t=coefficients)
        else:
            cross = AttnProcessor2_0()
        for name, module in attention_modules(pipe.unet):
            if name.endswith("attn1"):
                module.set_processor(
                    ConfidenceProcessor(
                        original_bank.memory,
                        aligned_bank.memory,
                        name,
                        coefficients,
                        gates,
                        beta,
                        fused,
                        cache,
                    )
                    if active
                    else AttnProcessor2_0()
                )
            else:
                module.set_processor(cross)
        conditional = pipe.unet(latent, timestep, encoder_hidden_states=text).sample
        original_bank.memory.clear()
        aligned_bank.memory.clear()
        for _, module in attention_modules(pipe.unet):
            module.set_processor(AttnProcessor2_0())
        unconditional = pipe.unet(
            latent, timestep, encoder_hidden_states=empty_text
        ).sample
        prediction = unconditional + pipe.guidance_scale * (conditional - unconditional)
        latent = pipe.scheduler.step(
            sample=latent, model_output=prediction, timestep=timestep
        ).prev_sample
    return latent, {
        "x".join(map(str, k[:2])): float(v.float().mean()) for k, v in cache.items()
    }


@dataclass
class ReferenceInputs:
    original: ReferenceCollector
    aligned: ReferenceCollector
    gates: torch.Tensor
    fields: dict


def build_reference_inputs(source, target, correspondence, frames=7):
    original = ReferenceCollector(source, target, frames)
    aligned = ReferenceCollector(source, target, frames)
    one = torch.ones(source.shape[-2:], device=source.device, dtype=torch.float32)
    target_in_source = eq8_transport(
        target, correspondence.source_to_target, correspondence.source_reliability
    )
    source_in_target = eq8_transport(
        source, correspondence.target_to_source, correspondence.target_reliability
    )
    for i in range(frames):
        original.capture(i, source, target, one)
        if i in (0, frames - 1):
            first, second = source, target
        elif i <= frames // 2:
            first, second = source, target_in_source
        else:
            first, second = source_in_target, target
        aligned.capture(i, first, second, one)
    gates, fields = confidence_gates(correspondence, frames)
    return ReferenceInputs(original, aligned, gates, fields)
