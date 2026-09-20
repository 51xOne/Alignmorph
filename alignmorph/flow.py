"""Regularize semantic correspondence before latent transport."""

import math
from typing import Dict, Tuple
import torch
from .correspondence import Correspondence
from .geometry import identity_map, sample_scalar
from .flow_fit import ConsolidatedFlow, consolidate_point_map
from .flow_geometry import (
    _gaussian_blur,
    _sample_point_map,
    _cell_determinants,
    repair_folded_point_map,
)


def _tensor_smoothstep(values: torch.Tensor, start: float, full: float) -> torch.Tensor:
    if not math.isfinite(start) or not math.isfinite(full) or not full > start:
        raise ValueError("smoothstep bounds must be finite and increasing")
    position = ((values - float(start)) / (float(full) - float(start))).clamp(0.0, 1.0)
    return position.square() * (3.0 - 2.0 * position)


def _semantic_trust_field(
    point_map: torch.Tensor,
    confidence: torch.Tensor,
    reverse_map: torch.Tensor,
    reverse_confidence: torch.Tensor,
    *,
    cycle_sigma_px: float = 3.0,
    coherence_sigma_px: float = 2.0,
    coherence_scale_px: float = 3.0,
    jacobian_blur_sigma_px: float = 1.5,
    trust_blur_sigma_px: float = 1.0,
    mix_start: float = 0.10,
    mix_full: float = 0.30,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Measure where a raw semantic map is locally usable as geometry.

    Confidence alone cannot distinguish a coherent part motion from a group of
    mutually crossing matches.  This input-only score therefore combines the
    paper confidence, reverse-cycle agreement, local displacement coherence,
    and orientation quality.  The returned mix is deliberately local: it is
    high only where the raw OT map can be retained without treating the whole
    image as one affine sheet.
    """

    if point_map.ndim != 3 or point_map.shape[-1] != 2:
        raise ValueError("point_map must have shape (H,W,2)")
    if confidence.shape != point_map.shape[:2]:
        raise ValueError("confidence must match point_map")
    if reverse_map.shape != point_map.shape:
        raise ValueError("reverse_map must match point_map")
    if reverse_confidence.shape != confidence.shape:
        raise ValueError("reverse_confidence must match confidence")

    observed = point_map.float()
    confidence_work = confidence.float()
    reverse_work = reverse_map.float()
    reverse_confidence_work = reverse_confidence.float()
    height, width = confidence.shape
    identity = identity_map(height, width, device=observed.device, dtype=observed.dtype)
    finite = torch.isfinite(observed).all(dim=-1) & torch.isfinite(confidence_work)
    in_bounds = (
        (observed[..., 0] >= -0.5)
        & (observed[..., 0] <= float(width) - 0.5)
        & (observed[..., 1] >= -0.5)
        & (observed[..., 1] <= float(height) - 0.5)
    )
    valid = finite & in_bounds & (confidence_work > 0.0)
    observed = torch.where(finite.unsqueeze(-1), observed, identity)
    reverse_finite = torch.isfinite(reverse_work).all(dim=-1) & torch.isfinite(
        reverse_confidence_work
    )
    reverse_work = torch.where(reverse_finite.unsqueeze(-1), reverse_work, identity)
    reverse_confidence_work = torch.where(
        reverse_finite,
        reverse_confidence_work.clamp_min(0.0),
        torch.zeros_like(reverse_confidence_work),
    )
    if not bool(valid.any()):
        zeros = torch.zeros_like(confidence_work)
        return (
            zeros,
            zeros,
            {
                "trust_mean": 0.0,
                "semantic_mix_mean": 0.0,
                "cycle_error_median_px": None,
                "local_coherence_mean": 0.0,
                "jacobian_quality_mean": 0.0,
            },
        )

    confidence_scale = torch.quantile(confidence_work[valid], 0.95).clamp_min(1e-12)
    normalized_confidence = (confidence_work / confidence_scale).clamp(0.0, 1.0)
    reverse_positive = reverse_confidence_work[reverse_confidence_work > 0.0]
    reverse_scale = (
        torch.quantile(reverse_positive, 0.95).clamp_min(1e-12)
        if reverse_positive.numel()
        else reverse_confidence_work.new_tensor(1.0)
    )
    cycled = _sample_point_map(reverse_work, observed)
    cycle_error = torch.linalg.vector_norm(cycled - identity, dim=-1)
    sampled_reverse_confidence = sample_scalar(
        reverse_confidence_work, observed, padding_mode="zeros"
    )
    cycle_reliability = (
        normalized_confidence
        * (sampled_reverse_confidence / reverse_scale).clamp(0.0, 1.0)
        * torch.exp(-0.5 * (cycle_error / float(cycle_sigma_px)).square())
    )
    cycle_reliability = torch.where(
        valid, cycle_reliability, torch.zeros_like(cycle_reliability)
    )
    cycle_quality = _tensor_smoothstep(cycle_reliability, 0.05, 0.35)

    displacement = observed - identity
    locally_smooth = _gaussian_blur(displacement, coherence_sigma_px)
    coherence_error = torch.linalg.vector_norm(displacement - locally_smooth, dim=-1)
    coherence = torch.exp(-0.5 * (coherence_error / float(coherence_scale_px)).square())

    cell_quality = _tensor_smoothstep(
        _cell_determinants(observed).amin(dim=-1), -0.05, 0.35
    )
    vertex_quality = torch.zeros_like(confidence_work)
    vertex_count = torch.zeros_like(confidence_work)
    for rows, columns in (
        (slice(None, -1), slice(None, -1)),
        (slice(1, None), slice(None, -1)),
        (slice(None, -1), slice(1, None)),
        (slice(1, None), slice(1, None)),
    ):
        vertex_quality[rows, columns] += cell_quality
        vertex_count[rows, columns] += 1.0
    vertex_quality = _gaussian_blur(
        vertex_quality / vertex_count.clamp_min(1.0),
        jacobian_blur_sigma_px,
    ).clamp(0.0, 1.0)

    trust = _gaussian_blur(
        cycle_quality * coherence * vertex_quality,
        trust_blur_sigma_px,
    ).clamp(0.0, 1.0)
    semantic_mix = _tensor_smoothstep(trust, mix_start, mix_full)
    diagnostics = {
        "trust_mean": float(trust.mean().item()),
        "trust_p90": float(torch.quantile(trust, 0.9).item()),
        "semantic_mix_mean": float(semantic_mix.mean().item()),
        "semantic_mix_p90": float(torch.quantile(semantic_mix, 0.9).item()),
        "cycle_error_median_px": float(cycle_error[valid].median().item()),
        "local_coherence_mean": float(coherence[valid].mean().item()),
        "jacobian_quality_mean": float(vertex_quality[valid].mean().item()),
        "parameters": {
            "cycle_sigma_px": float(cycle_sigma_px),
            "coherence_sigma_px": float(coherence_sigma_px),
            "coherence_scale_px": float(coherence_scale_px),
            "jacobian_blur_sigma_px": float(jacobian_blur_sigma_px),
            "trust_blur_sigma_px": float(trust_blur_sigma_px),
            "mix_start": float(mix_start),
            "mix_full": float(mix_full),
        },
    }
    return trust, semantic_mix, diagnostics


def constrain_semantic_point_map(
    point_map: torch.Tensor,
    confidence: torch.Tensor,
    *,
    reverse_map: torch.Tensor,
    reverse_confidence: torch.Tensor,
    highband_reliability_mode: str = "hybrid",
) -> ConsolidatedFlow:
    """Retain coherent raw OT motion over a topology-safe scaffold.

    Unlike :func:`consolidate_point_map`, the raw semantic field is not reduced
    to a globally smooth residual everywhere.  A local, input-only trust map
    chooses between the raw OT geometry and the topology-safe scaffold.  The
    resulting endpoint map is repaired only where its Jacobian still folds.
    By default the same mixture also builds the high-frequency support used by
    Eq. 8.  ``correspondence`` instead returns the input Eq. 4 reliability as
    the high-band support after point-map repair.
    """

    if highband_reliability_mode not in ("hybrid", "correspondence"):
        raise ValueError(
            "highband_reliability_mode must be 'hybrid' or 'correspondence'"
        )

    scaffold = consolidate_point_map(
        point_map,
        confidence,
        reverse_map=reverse_map,
        reverse_confidence=reverse_confidence,
    )
    original_dtype = point_map.dtype
    observed = point_map.float()
    height, width = observed.shape[:2]
    identity = identity_map(height, width, device=observed.device, dtype=observed.dtype)
    finite = torch.isfinite(observed).all(dim=-1)
    in_bounds = (
        (observed[..., 0] >= -0.5)
        & (observed[..., 0] <= float(width) - 0.5)
        & (observed[..., 1] >= -0.5)
        & (observed[..., 1] <= float(height) - 0.5)
    )
    observed = torch.where((finite & in_bounds).unsqueeze(-1), observed, identity)
    trust, semantic_mix, trust_diagnostics = _semantic_trust_field(
        observed,
        confidence,
        reverse_map,
        reverse_confidence,
    )
    scaffold_map = scaffold.point_map.float()
    candidate = scaffold_map + semantic_mix.unsqueeze(-1) * (observed - scaffold_map)
    constrained, repair_diagnostics = repair_folded_point_map(
        candidate,
        jacobian_floor=0.05,
        smoothing_sigma_px=2.0,
        mask_sigma_px=1.0,
        relaxation=0.4,
    )

    valid_confidence = torch.isfinite(confidence) & (confidence > 0.0)
    if bool(valid_confidence.any()):
        confidence_scale = torch.quantile(
            confidence.float()[valid_confidence], 0.95
        ).clamp_min(1e-12)
        raw_support = (confidence.float() / confidence_scale).clamp(0.0, 1.0)
    else:
        raw_support = torch.zeros_like(confidence, dtype=torch.float32)
    if highband_reliability_mode == "correspondence":
        support = torch.where(
            torch.isfinite(confidence),
            confidence.float(),
            torch.zeros_like(confidence, dtype=torch.float32),
        ).clamp(0.0, 1.0)
    else:
        support = (
            semantic_mix * raw_support + (1.0 - semantic_mix) * scaffold.support.float()
        ).clamp(0.0, 1.0)

    raw_error = torch.linalg.vector_norm(constrained.float() - observed, dim=-1)
    high_trust = semantic_mix >= 0.999
    diagnostics = {
        "regularization": "topology_constrained_semantic_ot_v1",
        "composition": "convex_raw_ot_over_topology_scaffold",
        "scaffold": scaffold.diagnostics,
        "trust": trust_diagnostics,
        "repair": repair_diagnostics,
        "highband_reliability_mode": highband_reliability_mode,
        "support_mean": float(support.mean().item()),
        "cycle_error_median_px": trust_diagnostics["cycle_error_median_px"],
        "raw_retention_error_mean_px": float(raw_error.mean().item()),
        "raw_retention_error_high_trust_mean_px": (
            float(raw_error[high_trust].mean().item())
            if bool(high_trust.any())
            else None
        ),
        "high_trust_fraction": float(high_trust.float().mean().item()),
    }
    return ConsolidatedFlow(
        point_map=constrained.to(dtype=original_dtype),
        support=support.to(dtype=original_dtype),
        diagnostics=diagnostics,
    )


def constrain_semantic_correspondence(
    correspondence: Correspondence,
    *,
    highband_reliability_mode: str = "hybrid",
) -> Tuple[Correspondence, Dict[str, object]]:
    """Build the same topology-constrained semantic field in both directions."""

    forward = constrain_semantic_point_map(
        correspondence.source_to_target,
        correspondence.source_reliability,
        reverse_map=correspondence.target_to_source,
        reverse_confidence=correspondence.target_reliability,
        highband_reliability_mode=highband_reliability_mode,
    )
    reverse = constrain_semantic_point_map(
        correspondence.target_to_source,
        correspondence.target_reliability,
        reverse_map=correspondence.source_to_target,
        reverse_confidence=correspondence.source_reliability,
        highband_reliability_mode=highband_reliability_mode,
    )
    return (
        Correspondence(
            source_to_target=forward.point_map,
            target_to_source=reverse.point_map,
            source_reliability=forward.support,
            target_reliability=reverse.support,
        ),
        {
            "source_to_target": forward.diagnostics,
            "target_to_source": reverse.diagnostics,
        },
    )
