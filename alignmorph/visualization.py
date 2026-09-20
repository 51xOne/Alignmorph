"""Inspect latent transport, its frequency terms, and the actual draft operands."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import AutoencoderKL
from .sampling import load_im_from_path
from .flow import constrain_semantic_correspondence
from .refinement import load_correspondence
from .transport import (
    transport_components,
    draft_operands,
    build_draft_path,
    _partial_map,
)
from .io import sha256
from .memory import confidence_gates, MAX_ALIGNED_WEIGHT


def rgb_image(tensor):
    array = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return Image.fromarray((array.clip(0, 1) * 255).round().astype(np.uint8))


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def panel(images, labels, columns=2):
    width, height = images[0].size
    heading = 52
    canvas = Image.new(
        "RGB",
        (
            columns * width,
            ((len(images) + columns - 1) // columns) * (height + heading),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels)):
        x = i % columns * width
        y = i // columns * (height + heading)
        draw.text((x + 10, y + 10), label, font=font(23), fill="black")
        canvas.paste(im, (x, y + heading))
    return canvas


def signed_channels(tensor, scale, side=768):
    """Four latent channels, blue=negative, white=zero, red=positive."""
    x = tensor.detach().float().cpu().squeeze(0).numpy()
    if x.shape[0] != 4:
        raise ValueError("Expected four latent channels")
    out = Image.new("RGB", (side, side), "white")
    d = ImageDraw.Draw(out)
    for c in range(4):
        value = np.clip(x[c] / max(float(scale), 1e-8), -1, 1)
        rgb = np.stack(
            [1 - np.maximum(-value, 0), 1 - np.abs(value), 1 - np.maximum(value, 0)],
            axis=-1,
        )
        im = Image.fromarray((rgb * 255).round().astype(np.uint8)).resize(
            (side // 2, side // 2), Image.Resampling.NEAREST
        )
        a = c % 2 * side // 2
        b = c // 2 * side // 2
        out.paste(im, (a, b))
        d.rectangle((a, b, a + 48, b + 28), fill="white")
        d.text((a + 4, b + 2), f"C{c}", font=font(20), fill="black")
    return out


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_cpu(v) for v in value]
    return value


class WarpVisualizer:
    def __init__(self, model, device="cuda"):
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.vae = (
            AutoencoderKL.from_pretrained(model, subfolder="vae")
            .to(self.device, self.dtype)
            .eval()
        )

    def decode(self, latent):
        return rgb_image(
            (
                self.vae.decode(latent / self.vae.config.scaling_factor).sample / 2
                + 0.5
            ).clamp(0, 1)
        )

    @torch.no_grad()
    def render(
        self, pair, correspondence_path, output, *, latents_path=None, strength=1.0
    ):
        if not 0 <= strength <= 1:
            raise ValueError("strength must lie in [0,1]")
        output = Path(output) / pair["exp_id"]
        if output.exists():
            raise FileExistsError(f"{output} exists; select a new output directory")
        hashes = [sha256(p) for p in pair["image_paths"]]
        corr_hash = sha256(correspondence_path)
        images = [
            load_im_from_path(p, [768, 768]).to(self.device, self.dtype)
            for p in pair["image_paths"]
        ]
        saved = None
        if latents_path is None:
            endpoints = [
                self.vae.encode(im).latent_dist.mean * self.vae.config.scaling_factor
                for im in images
            ]
            encoding = "posterior_mean"
        else:
            saved = torch.load(latents_path, map_location="cpu", weights_only=True)
            if (
                saved.get("input_sha256") != hashes
                or saved.get("correspondence_sha256") != corr_hash
            ):
                raise ValueError(
                    "Saved latents do not match this image pair and correspondence"
                )
            endpoints = [x.to(self.device, self.dtype) for x in saved["endpoints"]]
            encoding = "saved_sampled_endpoints"
        source, target = endpoints
        corr = load_correspondence(correspondence_path, source.shape[-2:], self.device)
        corr, flow = constrain_semantic_correspondence(
            corr, highband_reliability_mode="correspondence"
        )
        kv_gates, kv_fields = confidence_gates(corr, 7)
        maps = [corr.target_to_source, corr.source_to_target]
        if strength != 1.0:
            maps = [_partial_map(m, strength) for m in maps]
        weights = [corr.target_reliability, corr.source_reliability]
        terms = [
            transport_components(x, m, w) for x, m, w in zip(endpoints, maps, weights)
        ]
        operands, middle = draft_operands(source, target, corr)
        draft = build_draft_path(source, target, corr).latents
        for x in endpoints + [draft] + [v for t in terms for v in t.values()]:
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite visualization tensor")
        output.mkdir(parents=True)
        rendered = []
        labels = []
        for key, label in [
            ("original", "VAE reconstruction"),
            ("full_warp", "Full latent warp W(z)"),
            ("transported", "Multiband transport"),
        ]:
            for direction, t in zip(
                ["SRC into TGT layout", "TGT into SRC layout"], terms
            ):
                rendered.append(self.decode(t[key]))
                labels.append(
                    f"{label}: "
                    + (direction.split(" into ")[0] if key == "original" else direction)
                )
        panel(rendered, labels).save(output / "latent.png")
        lows = []
        labels = []
        for key, label in [
            ("low", "Original low-frequency latent"),
            ("warped_low", "Warped low-frequency latent"),
        ]:
            for name, t in zip(["SRC", "TGT"], terms):
                lows.append(self.decode(t[key]))
                labels.append(f"{name}: {label} (decoded)")
        panel(lows, labels).save(output / "low_frequency.png")
        high_keys = ["high", "warped_high", "moved_high", "retained_high"]
        scale = max(
            float(
                torch.quantile(
                    torch.cat(
                        [t[k].float().flatten() for t in terms for k in high_keys]
                    ),
                    0.995,
                ).abs()
            ),
            float(
                torch.quantile(
                    torch.cat(
                        [t[k].float().flatten() for t in terms for k in high_keys]
                    ),
                    0.005,
                ).abs()
            ),
            1e-8,
        )
        bands = []
        labels = []
        for key, label in [
            ("high", "H"),
            ("warped_high", "W(H)"),
            ("moved_high", "w * W(H)"),
            ("retained_high", "(1-w) * H"),
        ]:
            for name, t in zip(["SRC", "TGT"], terms):
                bands.append(signed_channels(t[key], scale))
                labels.append(f"{name}: {label}; scale +/-{scale:.3g}")
        panel(bands, labels).save(output / "high_frequency.png")
        paths = []
        labels = []
        for i, x in enumerate(operands):
            names = ["Draft", "SRC operand", "TGT operand"]
            if x["phase"] == "midpoint":
                names = [
                    "Midpoint draft",
                    "Source-chart blend",
                    "Transported target-chart blend",
                ]
            for name, value in zip(names, [draft[i : i + 1], x["first"], x["second"]]):
                paths.append(self.decode(value))
                labels.append(f'alpha={x["alpha"]:.3f}: {name}')
        panel(paths, labels, columns=3).save(output / "path.png")
        panel(
            [self.decode(middle[k]) for k in ["left", "right", "transported_right"]]
            + [self.decode(draft[3:4])],
            [
                "Source-chart blend (g=0.75)",
                "Target-chart blend (g=0.75)",
                "Target blend after half displacement",
                "Midpoint Slerp",
            ],
        ).save(output / "midpoint.png")
        reliability = []
        for w in weights:
            a = (
                (w.detach().float().cpu().numpy().clip(0, 1) * 255)
                .round()
                .astype(np.uint8)
            )
            reliability.append(
                Image.fromarray(a)
                .convert("RGB")
                .resize((768, 768), Image.Resampling.NEAREST)
            )
        panel(
            reliability, ["SRC transport reliability", "TGT transport reliability"]
        ).save(output / "reliability.png")
        gate_images, gate_labels = [], []
        for name, value in kv_fields.items():
            array = (
                (value.detach().float().cpu().numpy().clip(0, 1) * 255)
                .round()
                .astype(np.uint8)
            )
            gate_images.append(
                Image.fromarray(array)
                .convert("RGB")
                .resize((768, 768), Image.Resampling.NEAREST)
            )
            gate_labels.append(name.replace("_", " ") + " (0=black, 1=white)")
        panel(gate_images, gate_labels).save(output / "confidence_gates.png")
        midpoint_terms = transport_components(
            middle["right"], middle["half_map"], corr.source_reliability
        )
        torch.save(
            to_cpu(
                dict(
                    endpoints=endpoints,
                    transport=terms,
                    draft=draft,
                    operands=operands,
                    midpoint=middle,
                    midpoint_transport=midpoint_terms,
                    confidence_gates=kv_gates,
                    confidence_fields=kv_fields,
                    aligned_reference_weight=MAX_ALIGNED_WEIGHT * kv_gates,
                )
            ),
            output / "tensors.pt",
        )
        record = dict(
            pair=pair["exp_id"],
            input_sha256=hashes,
            correspondence_sha256=corr_hash,
            encoding=encoding,
            endpoint_display_displacement=strength,
            midpoint_displacement=0.75,
            outer_displacement=1.0,
            lowpass_kernel=9,
            aligned_reference_max_weight=MAX_ALIGNED_WEIGHT,
            confidence_thresholds=[0.1, 0.2],
            interior_aligned_weight_mean=float(
                (MAX_ALIGNED_WEIGHT * kv_gates[1:-1]).float().mean()
            ),
            high_band_color_scale=scale,
            high_band_color="blue negative; white zero; red positive; shared scale; tails clipped",
            decoder_note="Decoded low-frequency latents are illustrative; VAE decoding is nonlinear. High-frequency plots show latent channels, not decoded RGB.",
            path_note="At alpha=0.5 the two operands are blends, not individual warped endpoints.",
            flow=flow,
        )
        if saved is not None:
            record["draft_matches_saved_exactly"] = bool(
                torch.equal(draft.cpu(), saved["draft"])
            )
        (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")
        names = [
            "latent",
            "low_frequency",
            "high_frequency",
            "path",
            "midpoint",
            "reliability",
            "confidence_gates",
        ]
        body = "".join(
            f'<h2>{n.replace("_"," ")}</h2><a href="{n}.png"><img src="{n}.png" loading="lazy"></a>'
            for n in names
        )
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>AlignMorph warping</title><style>body{max-width:1500px;margin:30px auto;font:16px system-ui}img{width:100%}</style><h1>'
            + pair["exp_id"]
            + '</h1><p>Latent transport before diffusion. At the midpoint, path operands are the two blends. High-band plots show four signed latent channels with one shared scale.</p><p><a href="metadata.json">Metadata</a> · <a href="tensors.pt">Raw tensors</a></p>'
            + body
        )
        return record
