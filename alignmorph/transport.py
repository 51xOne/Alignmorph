"""Multiband transport and the bi-phase draft path."""

from dataclasses import dataclass

from typing import List, Dict, Tuple

import torch

import torch.nn.functional as F

from .geometry import identity_map, pixel_to_normalized, sample_field, sample_scalar


@dataclass(frozen=True)
class LatentPath:
    latents: torch.Tensor
    diagnostics: List[Dict[str, object]]


def eq8_transport(
    latent, pullback_map, reliability, kernel_size=9, padding_mode="border"
):
    """Warp structure; retain unreliable high-frequency content in place."""
    low, high = split_multiband(latent, kernel_size)
    point_map = _partial_map(pullback_map, 1.0)
    warped_low = _warp(low, point_map, padding_mode=padding_mode)
    warped_high = _warp(high, point_map, padding_mode=padding_mode)
    weight = reliability.to(device=latent.device, dtype=latent.dtype).clamp(0.0, 1.0)
    weight = weight.unsqueeze(0).unsqueeze(0)
    return warped_low + weight * warped_high + (1.0 - weight) * high


def split_multiband(
    latent: torch.Tensor, kernel_size: int = 9
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``latent`` using the paper's replicate-padded average low-pass."""

    if latent.ndim != 4:
        raise ValueError("latent must have shape (B,C,H,W)")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if kernel_size == 1:
        low = latent
    else:
        pad = kernel_size // 2
        low = F.avg_pool2d(
            F.pad(latent, (pad, pad, pad, pad), mode="replicate"),
            kernel_size=kernel_size,
            stride=1,
        )
    return low, latent - low


def compensate_latent_detail(
    latents: torch.Tensor,
    *,
    gain: float = 1.0,
    kernel_size: int = 9,
) -> torch.Tensor:
    """Rescale the residual latent band before VAE decoding.

    A symmetric temporal envelope keeps the two endpoint latents exact and
    applies the largest compensation near the midpoint.  This operation does
    not resample pixels or decoded RGB images.
    """

    if latents.ndim != 4 or latents.shape[0] < 3:
        raise ValueError("latents must have shape (N,C,H,W) with N >= 3")
    if not isinstance(gain, (int, float)) or not 1.0 <= float(gain) <= 2.0:
        raise ValueError("gain must be in [1,2]")
    if float(gain) == 1.0:
        return latents
    low, high = split_multiband(latents.float(), kernel_size=kernel_size)
    alpha = torch.linspace(
        0.0,
        1.0,
        latents.shape[0],
        device=latents.device,
        dtype=torch.float32,
    )
    frame_gain = 1.0 + (float(gain) - 1.0) * torch.sin(torch.pi * alpha).square()
    compensated = low + frame_gain.view(-1, 1, 1, 1) * high
    compensated = compensated.to(dtype=latents.dtype)
    compensated[0].copy_(latents[0])
    compensated[-1].copy_(latents[-1])
    return compensated


def _warp(
    tensor: torch.Tensor,
    pullback_map: torch.Tensor,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    if tensor.ndim != 4 or pullback_map.shape != (*tensor.shape[-2:], 2):
        raise ValueError("tensor and pullback_map have incompatible shapes")
    grid = (
        pixel_to_normalized(
            pullback_map, tensor.shape[-2], tensor.shape[-1], align_corners=False
        )
        .unsqueeze(0)
        .to(device=tensor.device, dtype=tensor.dtype)
    )
    if tensor.shape[0] > 1:
        grid = grid.expand(tensor.shape[0], -1, -1, -1)
    return F.grid_sample(
        tensor,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )


def _partial_map(point_map: torch.Tensor, amount: float) -> torch.Tensor:
    if not 0.0 <= amount <= 1.0:
        raise ValueError("amount must be in [0,1]")
    identity = identity_map(
        *point_map.shape[:2], device=point_map.device, dtype=point_map.dtype
    )
    return identity + float(amount) * (point_map - identity)


def _slerp(
    first: torch.Tensor,
    second: torch.Tensor,
    alpha: float,
    threshold: float = 0.9995,
) -> torch.Tensor:
    """Match the released FreeMorph row-wise Slerp exactly.

    FreeMorph normalizes along the final latent dimension rather than
    flattening a complete sample.  Preserving that convention is essential:
    a global flattened Slerp changes the baseline itself and produces an
    averaged, visibly softer path before AlignMorph transport is applied.
    """

    if first.shape != second.shape:
        raise ValueError("Slerp endpoints must have identical shapes")
    first_norm = torch.norm(first, dim=-1)
    second_norm = torch.norm(second, dim=-1)
    first_unit = first / first_norm.unsqueeze(-1)
    second_unit = second / second_norm.unsqueeze(-1)
    dot = (first_unit * second_unit).sum(-1)
    dot_magnitude = dot.abs()
    use_linear = dot_magnitude.isnan() | (dot_magnitude > float(threshold))
    use_spherical = ~use_linear
    output = torch.zeros_like(first)
    if bool(use_linear.any()):
        linear = torch.lerp(first, second, float(alpha))
        output = linear.where(use_linear.unsqueeze(-1), output)
    if bool(use_spherical.any()):
        angle = dot.arccos().unsqueeze(-1)
        sine = angle.sin()
        angle_t = angle * float(alpha)
        weight_first = (angle - angle_t).sin() / sine
        weight_second = angle_t.sin() / sine
        spherical = weight_first * first + weight_second * second
        output = spherical.where(use_spherical.unsqueeze(-1), output)
    return output


def transport_components(latent, pullback_map, reliability, kernel_size=9):
    """Expose the exact low/high terms of the multiband transport operator."""
    low, high = split_multiband(latent, kernel_size)
    mapping = _partial_map(pullback_map, 1.0)
    warped_low, warped_high = _warp(low, mapping), _warp(high, mapping)
    weight = reliability.to(latent).clamp(0, 1)[None, None]
    moved_high = weight * warped_high
    retained_high = (1.0 - weight) * high
    return dict(
        original=latent,
        low=low,
        high=high,
        warped_low=warped_low,
        warped_high=warped_high,
        moved_high=moved_high,
        retained_high=retained_high,
        transported=warped_low + moved_high + retained_high,
        full_warp=_warp(latent, mapping),
        reliability=weight,
        mapping=mapping,
    )


def draft_operands(source, target, correspondence, frames=7, midpoint_strength=0.75):
    """Return the two actual Slerp operands of every frame, including midpoint blends.

    A pullback map samples the opposite endpoint in the current endpoint's chart.
    Only the midpoint uses reduced displacement. Its operands are two blends,
    not independently warped source and target images.
    """
    if frames < 3 or frames % 2 != 1:
        raise ValueError("The bi-phase path requires an odd frame count >= 3")
    if source.shape != target.shape or source.ndim != 4 or source.shape[0] != 1:
        raise ValueError("Expected matching single-image endpoint latents")
    if not 0 <= midpoint_strength <= 1:
        raise ValueError("midpoint_strength must be in [0, 1]")
    st, ts = correspondence.source_to_target, correspondence.target_to_source
    ws, wt = correspondence.source_reliability, correspondence.target_reliability
    target_in_source = eq8_transport(target, st, ws)
    source_in_target = eq8_transport(source, ts, wt)
    # Use the same pixel-coordinate arithmetic as the transport convention.
    identity = identity_map(*source.shape[-2:], device=source.device, dtype=st.dtype)
    partial_st = identity + midpoint_strength * (st - identity)
    partial_ts = identity + midpoint_strength * (ts - identity)
    partial_target = eq8_transport(target, partial_st, ws)
    partial_source = eq8_transport(source, partial_ts, wt)
    left = _slerp(source, partial_target, 0.5)
    right = _slerp(partial_source, target, 0.5)
    half_map = identity + 0.5 * (partial_st - identity)
    moved_right = eq8_transport(right, half_map, ws)
    operands = []
    for index in range(frames):
        alpha = index / (frames - 1)
        if index < frames // 2:
            a, b = source, target_in_source
            phase = "source"
            strength = 1.0
        elif index > frames // 2:
            a, b = source_in_target, target
            phase = "target"
            strength = 1.0
        else:
            a, b = left, moved_right
            phase = "midpoint"
            strength = midpoint_strength
        operands.append(
            dict(
                first=a,
                second=b,
                alpha=alpha,
                phase=phase,
                displacement_strength=strength,
            )
        )
    return operands, dict(
        left=left, right=right, transported_right=moved_right, half_map=half_map
    )


def build_draft_path(source, target, correspondence, frames=7, midpoint_strength=0.75):
    """Construct the bi-phase path with a reduced-displacement midpoint."""
    operands, _ = draft_operands(
        source, target, correspondence, frames, midpoint_strength
    )
    latents = torch.cat([_slerp(x["first"], x["second"], x["alpha"]) for x in operands])
    latents[0].copy_(source[0])
    latents[-1].copy_(target[0])
    diagnostics = [
        dict(
            index=i,
            alpha=x["alpha"],
            phase=x["phase"],
            displacement_strength=x["displacement_strength"],
        )
        for i, x in enumerate(operands)
    ]
    return LatentPath(latents, diagnostics)
