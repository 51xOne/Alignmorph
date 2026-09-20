"""Grid diagnostics, smoothing, and local fold repair."""

import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from .geometry import identity_map, sample_field


def _gaussian_kernel(sigma: float, reference: torch.Tensor) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(
        -radius, radius + 1, device=reference.device, dtype=reference.dtype
    )
    kernel = torch.exp(-0.5 * (x / float(sigma)).square())
    return kernel / kernel.sum()


def _gaussian_blur(values: torch.Tensor, sigma: float) -> torch.Tensor:
    if values.ndim not in (2, 3):
        raise ValueError("Gaussian blur expects (H,W) or (H,W,C)")
    channels = 1 if values.ndim == 2 else values.shape[-1]
    tensor = values.unsqueeze(-1) if values.ndim == 2 else values
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    kernel = _gaussian_kernel(sigma, tensor)
    radius = kernel.numel() // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    tensor = F.conv2d(
        F.pad(tensor, (radius, radius, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    tensor = F.conv2d(
        F.pad(tensor, (0, 0, radius, radius), mode="replicate"),
        vertical,
        groups=channels,
    )
    output = tensor.squeeze(0).permute(1, 2, 0)
    return output[..., 0] if values.ndim == 2 else output


def _sample_point_map(point_map: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Sample displacement so border extension preserves translations."""

    identity = identity_map(
        *point_map.shape[:2], device=point_map.device, dtype=point_map.dtype
    )
    displacement = sample_field(point_map - identity, points, padding_mode="border")
    return points + displacement


def _cell_determinants(point_map: torch.Tensor) -> torch.Tensor:
    top_dx = point_map[:-1, 1:] - point_map[:-1, :-1]
    bottom_dx = point_map[1:, 1:] - point_map[1:, :-1]
    left_dy = point_map[1:, :-1] - point_map[:-1, :-1]
    right_dy = point_map[1:, 1:] - point_map[:-1, 1:]

    def determinant(dx, dy):
        return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]

    return torch.stack(
        (
            determinant(top_dx, left_dy),
            determinant(top_dx, right_dy),
            determinant(bottom_dx, left_dy),
            determinant(bottom_dx, right_dy),
        ),
        dim=-1,
    )


def _chart_minimum(point_map: torch.Tensor, samples: int = 21) -> float:
    identity = identity_map(
        *point_map.shape[:2], device=point_map.device, dtype=point_map.dtype
    )
    minimum = float("inf")
    for alpha in torch.linspace(0.0, 1.0, samples, device=point_map.device):
        chart = identity + alpha * (point_map - identity)
        minimum = min(minimum, float(_cell_determinants(chart).min().item()))
    return minimum


def repair_folded_point_map(
    point_map: torch.Tensor,
    *,
    jacobian_floor: float = 0.05,
    smoothing_sigma_px: float = 3.0,
    mask_sigma_px: float = 1.5,
    relaxation: float = 0.5,
    max_iterations: int = 160,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Locally diffuse an OT displacement around folded cells."""

    if point_map.ndim != 3 or point_map.shape[-1] != 2:
        raise ValueError("point_map must have shape (H,W,2)")
    if not math.isfinite(jacobian_floor) or jacobian_floor < 0.0:
        raise ValueError("jacobian_floor must be finite and non-negative")
    if smoothing_sigma_px <= 0.0 or mask_sigma_px <= 0.0:
        raise ValueError("fold-repair sigmas must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must be in (0,1]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    original_dtype = point_map.dtype
    observed = point_map.float()
    height, width = observed.shape[:2]
    identity = identity_map(height, width, device=observed.device, dtype=observed.dtype)
    finite = torch.isfinite(observed).all(dim=-1)
    observed = torch.where(finite.unsqueeze(-1), observed, identity)
    displacement = observed - identity
    raw_determinants = _cell_determinants(observed)
    raw_negative_fraction = float((raw_determinants < 0.0).float().mean().item())

    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        current = identity + displacement
        bad_cells = _cell_determinants(current).amin(dim=-1) < jacobian_floor
        if not bool(bad_cells.any()):
            converged = True
            break
        vertex_mask = torch.zeros(
            (height, width), device=observed.device, dtype=observed.dtype
        )
        bad = bad_cells.to(dtype=observed.dtype)
        vertex_mask[:-1, :-1] += bad
        vertex_mask[1:, :-1] += bad
        vertex_mask[:-1, 1:] += bad
        vertex_mask[1:, 1:] += bad
        vertex_mask = _gaussian_blur(
            (vertex_mask > 0.0).to(dtype=observed.dtype), mask_sigma_px
        ).clamp(0.0, 1.0)
        smooth_displacement = _gaussian_blur(displacement, smoothing_sigma_px)
        displacement = displacement + float(relaxation) * vertex_mask.unsqueeze(-1) * (
            smooth_displacement - displacement
        )

    repaired = identity + displacement
    repaired_determinants = _cell_determinants(repaired)
    endpoint_minimum = float(repaired_determinants.min().item())
    correction = torch.linalg.vector_norm(repaired - observed, dim=-1)
    diagnostics = {
        "regularization": "local_ot_fold_diffusion",
        "raw_negative_cell_fraction": raw_negative_fraction,
        "repaired_negative_cell_fraction": float(
            (repaired_determinants < 0.0).float().mean().item()
        ),
        "endpoint_min_jacobian_det": endpoint_minimum,
        "chart_path_min_jacobian_det": _chart_minimum(repaired),
        "iterations": iterations,
        "converged": converged or endpoint_minimum >= jacobian_floor,
        "jacobian_floor": float(jacobian_floor),
        "smoothing_sigma_px": float(smoothing_sigma_px),
        "mask_sigma_px": float(mask_sigma_px),
        "relaxation": float(relaxation),
        "correction_mean_px": float(correction.mean().item()),
        "correction_p90_px": float(torch.quantile(correction, 0.9).item()),
    }
    return repaired.to(dtype=original_dtype), diagnostics
