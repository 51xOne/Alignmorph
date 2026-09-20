"""Fit a stable global motion and a supported local residual."""

import math
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
from .geometry import identity_map, sample_scalar
from .flow_geometry import _gaussian_blur, _sample_point_map, _chart_minimum


@dataclass(frozen=True)
class ConsolidatedFlow:
    point_map: torch.Tensor
    support: torch.Tensor
    diagnostics: Dict[str, object]


def _weighted_affine(
    source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    design = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
    root_weight = weights.clamp_min(1e-12).sqrt().unsqueeze(1)
    return torch.linalg.lstsq(design * root_weight, target * root_weight).solution


def _robust_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    base_weights: torch.Tensor,
    iterations: int,
    huber_delta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = base_weights
    design = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
    affine = _weighted_affine(source, target, weights)
    errors = torch.linalg.vector_norm(design @ affine - target, dim=1)
    for _ in range(iterations):
        median = errors.median()
        mad = (errors - median).abs().median()
        scale = torch.maximum(1.4826 * mad, 0.25 * median).clamp_min(1e-4)
        cutoff = float(huber_delta) * scale
        robust = torch.where(
            errors <= cutoff,
            torch.ones_like(errors),
            cutoff / errors.clamp_min(1e-12),
        )
        weights = base_weights * robust
        affine = _weighted_affine(source, target, weights)
        errors = torch.linalg.vector_norm(design @ affine - target, dim=1)
    return affine, weights, errors


def _weighted_similarity(
    source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    weights = weights / weights.sum().clamp_min(1e-12)
    source_mean = (weights[:, None] * source).sum(dim=0)
    target_mean = (weights[:, None] * target).sum(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.transpose(0, 1) @ (weights[:, None] * target_centered)
    left, singular, right_t = torch.linalg.svd(covariance)
    correction = torch.eye(2, device=source.device, dtype=source.dtype)
    if torch.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    numerator = (singular * torch.diag(correction)).sum()
    denominator = (weights * source_centered.square().sum(dim=1)).sum().clamp_min(1e-12)
    scale = (numerator / denominator).clamp_min(1e-6)
    linear = scale * rotation
    translation = target_mean - source_mean @ linear
    return torch.cat((linear, translation.unsqueeze(0)), dim=0)


def consolidate_point_map(
    point_map: torch.Tensor,
    confidence: torch.Tensor,
    *,
    reverse_map: torch.Tensor = None,
    reverse_confidence: torch.Tensor = None,
    confidence_quantile: float = 0.65,
    confidence_power: float = 2.0,
    local_density_sigma_px: float = 2.0,
    local_density_floor: float = 0.12,
    cycle_sigma_px: float = 3.0,
    residual_sigma_px: float = 9.0,
    support_sigma_px: float = 13.0,
    residual_ridge: float = 0.03,
    local_residual_scale: float = 1.0,
    jacobian_floor: float = 0.1,
    safety_backtrack: float = 0.8,
    irls_iterations: int = 6,
    huber_delta: float = 2.5,
    min_anchors: int = 64,
) -> ConsolidatedFlow:
    """Fit a topology-safe point map using fixed global parameters."""

    if point_map.ndim != 3 or point_map.shape[-1] != 2:
        raise ValueError("point_map must have shape (H,W,2)")
    if confidence.shape != point_map.shape[:2]:
        raise ValueError("confidence must match point_map")
    if (reverse_map is None) != (reverse_confidence is None):
        raise ValueError("reverse_map and reverse_confidence must be provided together")
    if reverse_map is not None:
        if (
            reverse_map.shape != point_map.shape
            or reverse_confidence.shape != confidence.shape
        ):
            raise ValueError("reverse correspondence must match the forward lattice")
    if not 0.0 <= local_residual_scale <= 1.0:
        raise ValueError("local_residual_scale must be in [0,1]")
    original_dtype = point_map.dtype
    observed = point_map.double()
    confidence = confidence.double()
    height, width = confidence.shape
    identity = identity_map(height, width, device=observed.device, dtype=observed.dtype)

    def no_anchor_fallback(reason: str) -> ConsolidatedFlow:
        """Return the exact no-transport limit when a map has no usable evidence.

        A correspondence extractor can legitimately reject every match for a
        pair.  Treating that case as an exception makes a corpus evaluation
        brittle, while inventing motion would be worse.  Identity geometry and
        zero support are the deterministic limit of the same transport model.
        """

        return ConsolidatedFlow(
            point_map=identity.to(dtype=original_dtype),
            support=torch.zeros_like(confidence, dtype=original_dtype),
            diagnostics={
                "global_kind": "identity_no_anchor_fallback",
                "uses_reverse_cycle": reverse_map is not None,
                "fallback_reason": reason,
                "cycle_error_median_px": None,
                "cycle_error_p90_px": None,
                "selected_before_density": 0,
                "selected_anchors": 0,
                "density_filter_fallback": False,
                "residual_composition": "none",
                "requested_local_residual_scale": float(local_residual_scale),
                "effective_local_residual_scale": 0.0,
                "chart_path_min_jacobian_det": 1.0,
                "support_mean": 0.0,
            },
        )

    valid = (
        torch.isfinite(observed).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence > 0.0)
        & (observed[..., 0] >= -0.5)
        & (observed[..., 0] <= width - 0.5)
        & (observed[..., 1] >= -0.5)
        & (observed[..., 1] <= height - 0.5)
    )
    if int(valid.sum()) < min_anchors:
        return no_anchor_fallback("too_few_valid_correspondences")
    scale = torch.quantile(confidence[valid], 0.95).clamp_min(1e-12)
    reliability = (confidence / scale).clamp(0.0, 1.0)
    cycle_error_median = None
    cycle_error_p90 = None
    if reverse_map is not None:
        reverse_observed = reverse_map.double()
        reverse_conf = reverse_confidence.double()
        reverse_finite = torch.isfinite(reverse_observed).all(dim=-1) & torch.isfinite(
            reverse_conf
        )
        reverse_observed = torch.where(
            reverse_finite.unsqueeze(-1), reverse_observed, identity
        )
        reverse_conf = torch.where(
            reverse_finite, reverse_conf.clamp_min(0.0), torch.zeros_like(reverse_conf)
        )
        cycled = _sample_point_map(reverse_observed, observed)
        sampled_reverse_conf = sample_scalar(
            reverse_conf, observed, padding_mode="zeros"
        )
        reverse_positive = reverse_conf[reverse_conf > 0.0]
        reverse_scale = (
            torch.quantile(reverse_positive, 0.95).clamp_min(1e-12)
            if reverse_positive.numel()
            else reverse_conf.new_tensor(1.0)
        )
        cycle_error = torch.linalg.vector_norm(cycled - identity, dim=-1)
        cycle_weight = torch.exp(-0.5 * (cycle_error / float(cycle_sigma_px)).square())
        reliability = (
            reliability
            * (sampled_reverse_conf / reverse_scale).clamp(0.0, 1.0)
            * cycle_weight
        )
        valid = valid & torch.isfinite(cycle_error) & (sampled_reverse_conf > 0.0)
        if bool(valid.any()):
            cycle_error_median = float(cycle_error[valid].median().item())
            cycle_error_p90 = float(torch.quantile(cycle_error[valid], 0.9).item())
    reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
    positive = reliability[valid & (reliability > 0)]
    if positive.numel() < min_anchors:
        return no_anchor_fallback("too_few_cycle_weighted_correspondences")
    threshold = torch.quantile(positive, confidence_quantile)
    selected_before_density = valid & (reliability >= threshold)
    local_density = _gaussian_blur(
        selected_before_density.to(observed.dtype), local_density_sigma_px
    )
    selected = selected_before_density & (local_density >= local_density_floor)
    density_fallback = False
    if int(selected.sum()) < min_anchors:
        selected = selected_before_density
        density_fallback = True
    if int(selected.sum()) < min_anchors:
        return no_anchor_fallback("too_few_selected_correspondences")
    source = identity[selected]
    target = observed[selected]
    base_weights = reliability[selected].pow(confidence_power).clamp_min(1e-8)
    affine, robust_weights, affine_errors = _robust_affine(
        source, target, base_weights, irls_iterations, huber_delta
    )
    affine_linear = affine[:2]
    singular_values = torch.linalg.svdvals(affine_linear)
    affine_det = float(torch.linalg.det(affine_linear).item())
    affine_ratio = float(
        (singular_values.min() / singular_values.max().clamp_min(1e-12)).item()
    )
    global_kind = "robust_affine"
    if affine_det < jacobian_floor or affine_ratio < math.sqrt(jacobian_floor):
        similarity = _weighted_similarity(source, target, robust_weights)
        design = torch.cat((source, torch.ones_like(source[:, :1])), dim=1)
        similarity_errors = torch.linalg.vector_norm(
            design @ similarity - target, dim=1
        )
        similarity_det = float(torch.linalg.det(similarity[:2]).item())
        if similarity_det >= jacobian_floor and float(
            similarity_errors.median()
        ) <= float(affine_errors.median()):
            affine = similarity
            global_kind = "robust_similarity"
        elif affine_det < jacobian_floor:
            translation = (robust_weights[:, None] * (target - source)).sum(
                dim=0
            ) / robust_weights.sum().clamp_min(1e-12)
            affine = torch.zeros((3, 2), device=identity.device, dtype=identity.dtype)
            affine[0, 0] = 1.0
            affine[1, 1] = 1.0
            affine[2] = translation
            global_kind = "translation_fallback"
    design_grid = torch.cat((identity, torch.ones_like(identity[..., :1])), dim=-1)
    global_map = torch.einsum("hwk,kd->hwd", design_grid, affine)
    if _chart_minimum(global_map) < jacobian_floor:
        # The algebraic affine determinant can sit just above the threshold
        # while the sampled chart path falls just below it numerically.  The
        # topology contract is defined by the chart itself, so use the same
        # fixed translation fallback for every such case.
        translation = (robust_weights[:, None] * (target - source)).sum(
            dim=0
        ) / robust_weights.sum().clamp_min(1e-12)
        affine = torch.zeros((3, 2), device=identity.device, dtype=identity.dtype)
        affine[0, 0] = 1.0
        affine[1, 1] = 1.0
        affine[2] = translation
        global_kind = "translation_chart_safety_fallback"
        global_map = torch.einsum("hwk,kd->hwd", design_grid, affine)
        if _chart_minimum(global_map) < jacobian_floor:
            raise RuntimeError("translation fallback is not topology safe")

    dense_weights = torch.zeros_like(reliability)
    dense_weights[selected] = robust_weights
    observed_residual = observed - global_map
    numerator = _gaussian_blur(
        dense_weights.unsqueeze(-1) * observed_residual, residual_sigma_px
    )
    denominator = _gaussian_blur(dense_weights, residual_sigma_px)
    residual = numerator / (denominator.unsqueeze(-1) + residual_ridge)
    support_seed = dense_weights / dense_weights.max().clamp_min(1e-12)
    support = _gaussian_blur(support_seed, support_sigma_px)
    support = (support / support.max().clamp_min(1e-12)).clamp(0.0, 1.0)
    # Support is consumed by latent transport, so it is not multiplied into
    # the geometric residual a second time.
    local = torch.nan_to_num(residual)
    effective_scale = float(local_residual_scale)
    point_map_out = global_map + effective_scale * local
    for _ in range(64):
        if _chart_minimum(point_map_out) >= jacobian_floor:
            break
        effective_scale *= safety_backtrack
        if effective_scale < 1e-8:
            effective_scale = 0.0
        point_map_out = global_map + effective_scale * local
    minimum = _chart_minimum(point_map_out)
    if minimum < jacobian_floor:
        raise RuntimeError("local residual backtrack did not recover a safe flow")
    diagnostics = {
        "global_kind": global_kind,
        "uses_reverse_cycle": reverse_map is not None,
        "cycle_error_median_px": cycle_error_median,
        "cycle_error_p90_px": cycle_error_p90,
        "selected_before_density": int(selected_before_density.sum().item()),
        "selected_anchors": int(selected.sum().item()),
        "density_filter_fallback": density_fallback,
        "residual_composition": "global_plus_full_smooth_residual",
        "requested_local_residual_scale": float(local_residual_scale),
        "effective_local_residual_scale": effective_scale,
        "chart_path_min_jacobian_det": minimum,
        "support_mean": float(support.mean().item()),
    }
    return ConsolidatedFlow(
        point_map=point_map_out.to(dtype=original_dtype),
        support=support.to(dtype=original_dtype),
        diagnostics=diagnostics,
    )
