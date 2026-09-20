"""AlignMorph semantic correspondence (paper Eqs. 1--5).

This module consumes dense feature maps. Feature extraction is deliberately
kept behind a separate adapter so the OT mathematics can be tested without
downloading DINOv2 or compiling FeatUp.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .geometry import (
    identity_map,
    resize_point_map,
    resize_scalar,
    sample_field,
    sample_scalar,
)


@dataclass(frozen=True)
class Correspondence:
    """Bidirectional point maps and their Eq. 4 reliability fields."""

    source_to_target: torch.Tensor
    target_to_source: torch.Tensor
    source_reliability: torch.Tensor
    target_reliability: torch.Tensor

    def to_grids(
        self,
        source_hw: Tuple[int, int],
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> "Correspondence":
        """Project the correspondence to new source/target lattices (Eq. 5)."""

        if target_hw is None:
            target_hw = source_hw
        source_input_hw = tuple(self.source_to_target.shape[:2])
        target_input_hw = tuple(self.target_to_source.shape[:2])
        return Correspondence(
            source_to_target=resize_point_map(
                self.source_to_target,
                source_hw,
                target_input_hw=target_input_hw,
                target_output_hw=target_hw,
            ),
            target_to_source=resize_point_map(
                self.target_to_source,
                target_hw,
                target_input_hw=source_input_hw,
                target_output_hw=source_hw,
            ),
            source_reliability=resize_scalar(self.source_reliability, source_hw).clamp(
                0.0, 1.0
            ),
            target_reliability=resize_scalar(self.target_reliability, target_hw).clamp(
                0.0, 1.0
            ),
        )


def _flatten_features(features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if features.ndim != 3:
        raise ValueError("features must have shape (C,H,W)")
    channels, height, width = features.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("feature dimensions must be positive")
    flat = features.permute(1, 2, 0).reshape(height * width, channels).float()
    return F.normalize(flat, p=2, dim=-1), (height, width)


def _log_sinkhorn(
    cost: torch.Tensor, epsilon: float, iterations: int, marginal_epsilon: float = 0.0
) -> torch.Tensor:
    """Balanced entropic OT in the log domain (Eq. 1)."""

    if cost.ndim != 2 or cost.numel() == 0:
        raise ValueError("cost must be a non-empty matrix")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    rows, columns = cost.shape
    log_a = torch.full(
        (rows,), -math.log(float(rows)), device=cost.device, dtype=torch.float32
    )
    log_b = torch.full(
        (columns,),
        -math.log(float(columns)),
        device=cost.device,
        dtype=torch.float32,
    )
    if marginal_epsilon > 0:
        log_a = torch.log(torch.full_like(log_a, 1.0 / rows) + marginal_epsilon)
        log_b = torch.log(torch.full_like(log_b, 1.0 / columns) + marginal_epsilon)
    log_kernel = -cost.float() / float(epsilon)
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(int(iterations)):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    coupling = torch.exp(log_u[:, None] + log_kernel + log_v[None, :])
    if not bool(torch.isfinite(coupling).all()):
        raise FloatingPointError("Sinkhorn produced a non-finite coupling")
    return coupling


def _conditional(coupling: torch.Tensor, dim: int) -> torch.Tensor:
    normalizer = coupling.sum(dim=dim, keepdim=True)
    return coupling / normalizer.clamp_min(torch.finfo(coupling.dtype).tiny)


def _sharpness(conditional: torch.Tensor) -> torch.Tensor:
    """Map conditional row maxima from uniform=0 to one-hot=1."""

    columns = conditional.shape[1]
    if columns <= 1:
        return torch.ones(conditional.shape[0], device=conditional.device)
    row_max = conditional.max(dim=1).values
    return (
        torch.log((row_max * float(columns)).clamp_min(1.0)) / math.log(float(columns))
    ).clamp(0.0, 1.0)


def _coordinates(height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
    return identity_map(
        height, width, device=reference.device, dtype=reference.dtype
    ).reshape(-1, 2)


def _reliability_one_direction(
    forward_map: torch.Tensor,
    forward_sharpness: torch.Tensor,
    reverse_map: torch.Tensor,
    reverse_sharpness: torch.Tensor,
    *,
    cycle_threshold_px: float,
    cycle_sigma_px: float,
    minimum_confidence: float,
) -> torch.Tensor:
    height, width = forward_map.shape[:2]
    reverse_at_forward = sample_field(reverse_map, forward_map, padding_mode="zeros")
    reverse_confidence = sample_scalar(
        reverse_sharpness, forward_map, padding_mode="zeros"
    )
    source_points = identity_map(
        height, width, device=forward_map.device, dtype=forward_map.dtype
    )
    cycle_error = torch.linalg.vector_norm(reverse_at_forward - source_points, dim=-1)
    target_height, target_width = reverse_map.shape[:2]
    in_bounds = (
        (forward_map[..., 0] >= 0.0)
        & (forward_map[..., 0] <= float(target_width - 1))
        & (forward_map[..., 1] >= 0.0)
        & (forward_map[..., 1] <= float(target_height - 1))
    )
    valid = (
        in_bounds
        & torch.isfinite(forward_map).all(dim=-1)
        & torch.isfinite(cycle_error)
        & (cycle_error <= float(cycle_threshold_px))
        & (forward_sharpness >= float(minimum_confidence))
        & (reverse_confidence >= float(minimum_confidence))
    )
    cycle_weight = torch.exp(-cycle_error.square() / (2.0 * float(cycle_sigma_px) ** 2))
    reliability = forward_sharpness * reverse_confidence * cycle_weight
    return torch.where(valid, reliability, torch.zeros_like(reliability)).clamp(
        0.0, 1.0
    )


def project_conditional_coordinates(
    plan: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    radius: float = 4.0,
    row_chunk: int = 512,
) -> torch.Tensor:
    """Take a barycenter within a fixed Euclidean window around each OT peak."""
    if not math.isfinite(radius) or radius < 0 or row_chunk < 1:
        raise ValueError(
            "local projection requires a finite nonnegative radius and positive chunk"
        )
    if (
        plan.ndim != 2
        or coordinates.shape != (plan.shape[1], 2)
        or min(plan.shape) == 0
    ):
        raise ValueError("plan and target coordinates have incompatible shapes")
    centers = coordinates[plan.argmax(dim=1)]
    result = torch.empty_like(centers)
    tiny = torch.finfo(plan.dtype).tiny
    for start in range(0, plan.shape[0], row_chunk):
        stop = min(start + row_chunk, plan.shape[0])
        distance = coordinates[None] - centers[start:stop, None]
        weights = torch.where(
            distance.square().sum(-1) <= radius * radius,
            plan[start:stop],
            torch.zeros((), device=plan.device, dtype=plan.dtype),
        )
        mass = weights.sum(1, keepdim=True)
        result[start:stop] = torch.where(
            mass > 0, weights @ coordinates / mass.clamp_min(tiny), centers[start:stop]
        )
    return result


def dense_ot_correspondence(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    *,
    epsilon: float = 0.1,
    iterations: int = 100,
    max_tokens: int = 16_384,
    cycle_threshold_px: float = 8.0,
    cycle_sigma_px: float = 4.0,
    minimum_confidence: float = 0.2,
    projection_mode: str = "soft",
    local_soft_radius: float = 4.0,
    independent_reverse: bool = False,
    normalization_epsilon: float = 0.0,
    reliability_on_cpu: bool = False,
) -> Correspondence:
    """Compute bidirectional OT barycentric maps and cycle reliability.

    Args:
        source_features: DINOv2+FeatUp features shaped ``(C,Hs,Ws)``.
        target_features: DINOv2+FeatUp features shaped ``(C,Ht,Wt)``.
    """

    source, source_hw = _flatten_features(source_features)
    target, target_hw = _flatten_features(target_features)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature channel counts must match")
    if source.shape[0] > max_tokens or target.shape[0] > max_tokens:
        raise ValueError(
            "feature grid exceeds max_tokens; downsample features before OT"
        )
    if cycle_threshold_px <= 0 or cycle_sigma_px <= 0:
        raise ValueError("cycle threshold and sigma must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in [0,1]")
    if projection_mode not in ("soft", "local_soft"):
        raise ValueError("projection_mode must be soft or local_soft")
    if not math.isfinite(normalization_epsilon) or normalization_epsilon < 0:
        raise ValueError("normalization_epsilon must be finite and nonnegative")

    cost = 1.0 - source @ target.transpose(0, 1)
    coupling = _log_sinkhorn(cost, epsilon, iterations, normalization_epsilon)

    def conditional(plan):
        if normalization_epsilon == 0:
            return _conditional(plan, dim=1)
        return plan / (plan.sum(dim=1, keepdim=True) + normalization_epsilon)

    forward_conditional = conditional(coupling)
    if independent_reverse:
        reverse_cost = 1.0 - target @ source.transpose(0, 1)
        reverse_conditional = conditional(
            _log_sinkhorn(reverse_cost, epsilon, iterations, normalization_epsilon)
        )
    else:
        reverse_conditional = conditional(coupling.transpose(0, 1))
    target_coordinates = _coordinates(*target_hw, target)
    source_coordinates = _coordinates(*source_hw, source)

    def project(plan, coordinates, shape):
        points = (
            plan @ coordinates
            if projection_mode == "soft"
            else project_conditional_coordinates(
                plan, coordinates, radius=local_soft_radius
            )
        )
        return points.reshape(*shape, 2)

    source_to_target = project(forward_conditional, target_coordinates, source_hw)
    target_to_source = project(reverse_conditional, source_coordinates, target_hw)
    source_sharpness = _sharpness(forward_conditional).reshape(*source_hw)
    target_sharpness = _sharpness(reverse_conditional).reshape(*target_hw)
    reliability_fields = (
        source_to_target,
        source_sharpness,
        target_to_source,
        target_sharpness,
    )
    if reliability_on_cpu:
        reliability_fields = tuple(field.cpu() for field in reliability_fields)
    source_reliability = _reliability_one_direction(
        *reliability_fields,
        cycle_threshold_px=cycle_threshold_px,
        cycle_sigma_px=cycle_sigma_px,
        minimum_confidence=minimum_confidence,
    )
    target_reliability = _reliability_one_direction(
        *reliability_fields[2:],
        *reliability_fields[:2],
        cycle_threshold_px=cycle_threshold_px,
        cycle_sigma_px=cycle_sigma_px,
        minimum_confidence=minimum_confidence,
    )
    return Correspondence(
        source_to_target=source_to_target,
        target_to_source=target_to_source,
        source_reliability=source_reliability,
        target_reliability=target_reliability,
    )
