"""Encode endpoints, transport their layout, then refine with endpoint memories."""

import json
from pathlib import Path

import torch
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image

from .endpoint_attention import (
    FrameMeanAttention,
    EndpointMeanAttention,
    EndpointInterpolatedAttention,
)
from .sampling import (
    fourier_filter,
    generate_beta_tensor,
    linear_interpolation,
    load_im_from_path,
)
from .attention import (
    ReferenceBank,
    EndpointMemoryProcessor,
    ReferenceCollector,
    attention_modules,
    invert_references,
)
from .flow import constrain_semantic_correspondence
from .refinement import (
    load_correspondence,
    semantic_detail_gate,
)
from .transport import build_draft_path, compensate_latent_detail
from .io import sha256
from .memory import build_reference_inputs, denoise_mixed, MAX_ALIGNED_WEIGHT


class AlignMorphPipeline:
    """Generate a bi-phase morph with a reduced-displacement midpoint."""

    def __init__(self, model, device="cuda", seed=42):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Generation requires a CUDA device")
        self.seed = seed
        self.dtype = torch.float16
        self.steps, self.frames, self.resolution = 50, 7, 768
        self.guidance_scale = 7.5
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False, warn_only=True)
        set_seed(seed)
        torch.set_grad_enabled(False)
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae").to(
            self.device, self.dtype
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder"
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
        self.unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet").to(
            self.device, self.dtype
        )
        self.scheduler = DDIMScheduler.from_pretrained(model, subfolder="scheduler")
        self.scheduler.set_timesteps(self.steps)
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model, subfolder="scheduler"
        )
        self.inverse_scheduler.set_timesteps(self.steps)
        self.shared_noise = torch.randn(
            (1, 4, 96, 96), device=self.device, dtype=self.dtype
        )

    def encode(self, pair, generator):
        latents, texts, empty_texts, images = [], [], [], []
        for path, prompt in zip(pair["image_paths"], pair["prompts"]):
            image = load_im_from_path(path, [self.resolution, self.resolution]).to(
                self.device, self.dtype
            )
            images.append(image)
            ids = self.tokenizer(
                [prompt, ""], return_tensors="pt", truncation=True, padding="max_length"
            ).input_ids.to(self.device)
            text = self.text_encoder(ids).last_hidden_state.to(self.device, self.dtype)
            conditional, unconditional = text.chunk(2, dim=0)
            texts.append(conditional)
            empty_texts.append(unconditional)
            latents.append(
                self.vae.encode(image).latent_dist.sample(generator=generator)
                * self.vae.config.scaling_factor
            )
        return latents, texts, empty_texts, images

    def make_draft(self, endpoints, correspondence_path):
        correspondence = load_correspondence(
            correspondence_path, endpoints[0].shape[-2:], self.device
        )
        correspondence, flow = constrain_semantic_correspondence(
            correspondence, highband_reliability_mode="correspondence"
        )
        references = build_reference_inputs(*endpoints, correspondence, self.frames)
        path = build_draft_path(*endpoints, correspondence, frames=self.frames)
        return path, references, flow

    def invert_draft(self, latent, text, timesteps):
        iter_latent = latent.clone()
        for index, timestep in enumerate(timesteps):
            if index > int(self.steps * 0.3) and index < int(self.steps * 0.6):
                original = FrameMeanAttention()
            elif index > int(self.steps * 0.6):
                original = EndpointMeanAttention(is_fused=False)
            else:
                original = AttnProcessor2_0()
            for _, module in attention_modules(self.unet):
                module.set_processor(original)
            for _ in range(5):
                prediction = self.unet(
                    iter_latent, timestep, encoder_hidden_states=text
                ).sample
                iter_latent = self.inverse_scheduler.step(
                    sample=latent, model_output=prediction, timestep=timestep
                ).prev_sample
            latent = iter_latent.clone()
        return latent

    def denoise(
        self, latent, text, empty_text, timesteps, references, aligned_references, gates
    ):
        return denoise_mixed(
            self,
            latent,
            text,
            empty_text,
            timesteps,
            references,
            aligned_references,
            gates,
            MAX_ALIGNED_WEIGHT,
        )

    def diffusion_timesteps(self):
        forward = (self.scheduler.timesteps - 1)[10:]
        inverse = (self.inverse_scheduler.timesteps - 1)[:40]
        return forward, inverse

    def inject_noise(self, inverted, noise):
        thresholds = [44, 42, 42, 42, 42]
        interiors = [
            fourier_filter(
                x=inverted[i].unsqueeze(0), y=noise.clone(), threshold=threshold
            ).clone()
            for i, threshold in enumerate(thresholds, start=1)
        ]
        return torch.cat(
            [inverted[0].unsqueeze(0)] + interiors + [inverted[-1].unsqueeze(0)]
        )

    def prepare_references(self, collector, texts, forward_times):
        text = torch.cat([t.expand(self.frames, -1, -1) for t in texts])
        states = invert_references(
            self.unet, collector.latents(), text, self.scheduler, forward_times.flip(0)
        )
        return ReferenceBank(states, text, collector.reliability())

    def refine_detail(self, latent, flow):
        gate, confidence = semantic_detail_gate(
            flow, confidence_start=0.0, confidence_full=0.02
        )
        gain = 1.0 + gate * (1.1 - 1.0)
        if gain > 1.0:
            latent = compensate_latent_detail(latent, gain=gain, kernel_size=9)
        return latent, gain, confidence

    def save_images(self, latent, originals, output_path):
        images = []
        for frame in latent:
            decoded = self.vae.decode(
                frame.unsqueeze(0) / self.vae.config.scaling_factor
            ).sample
            images.append(
                (decoded / 2 + 0.5).clamp(0, 1).detach().to(torch.float).cpu()
            )
        images[0] = (originals[0] / 2 + 0.5).detach().to(torch.float).cpu()
        images[-1] = (originals[1] / 2 + 0.5).detach().to(torch.float).cpu()
        output_path = Path(output_path)
        save_image(torch.cat(images), str(output_path))

    @torch.no_grad()
    def generate(self, pair, correspondence_path, output_path, *, save_latents=False):
        endpoints, texts, empty_texts, originals = self.encode(pair, None)
        path, collector, flow = self.make_draft(endpoints, correspondence_path)
        text = linear_interpolation(*texts, self.frames).squeeze(0)
        empty = linear_interpolation(*empty_texts, self.frames).squeeze(0)

        forward_times, inverse_times = self.diffusion_timesteps()
        references = self.prepare_references(collector.original, texts, forward_times)
        inverted = self.invert_draft(path.latents.clone(), text, inverse_times)
        noisy = self.inject_noise(inverted, self.shared_noise)
        aligned_references = self.prepare_references(
            collector.aligned, texts, forward_times
        )
        refined, gate_means = self.denoise(
            noisy,
            text,
            empty,
            forward_times,
            references,
            aligned_references,
            collector.gates,
        )
        refined, gain, confidence = self.refine_detail(refined, flow)
        self.save_images(refined, originals, output_path)

        if save_latents:
            torch.save(
                dict(
                    endpoints=[x.detach().cpu() for x in endpoints],
                    draft=path.latents.detach().cpu(),
                    final=refined.detach().cpu(),
                    confidence_gates=collector.gates.detach().cpu(),
                    input_sha256=[sha256(p) for p in pair["image_paths"]],
                    correspondence_sha256=sha256(correspondence_path),
                ),
                Path(output_path).with_suffix(".pt"),
            )

        diagnostics = dict(
            pair=pair["exp_id"],
            chart_path=path.diagnostics,
            midpoint_displacement=0.75,
            outer_displacement=1.0,
            reference_memory="confidence_gated_original_and_aligned_endpoints",
            aligned_reference_max_weight=MAX_ALIGNED_WEIGHT,
            confidence_thresholds=[0.1, 0.2],
            interior_gate_mean=float(collector.gates[1:-1].float().mean()),
            interior_aligned_weight_mean=float(
                (MAX_ALIGNED_WEIGHT * collector.gates[1:-1]).float().mean()
            ),
            gate_means_by_resolution=gate_means,
            attention_stages=[8, 24, 40],
            lowpass_kernel=9,
            seed=self.seed,
            reference_weight_mean=[float(w.mean()) for w in references.reliability],
            detail_gain=gain,
            semantic_confidence=confidence,
        )
        Path(output_path).with_suffix(".json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
