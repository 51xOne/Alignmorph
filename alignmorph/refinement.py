"""Correspondence loading and latent-detail refinement."""

import math
import numpy as np
import torch
from .correspondence import Correspondence


def semantic_detail_gate(
    flow_diagnostics,
    *,
    confidence_start: float = 0.02,
    confidence_full: float = 0.15,
):
    """Derive a pair-global detail gate from bidirectional semantic trust."""

    start = float(confidence_start)
    full = float(confidence_full)
    if not 0.0 <= start < full <= 1.0:
        raise ValueError("semantic confidence thresholds are invalid")
    try:
        values = [
            float(flow_diagnostics[direction]["trust"]["semantic_mix_mean"])
            for direction in ("source_to_target", "target_to_source")
        ]
    except (KeyError, TypeError, ValueError):
        return 0.0, None
    if not all(math.isfinite(value) for value in values):
        return 0.0, None
    confidence = sum(values) / len(values)
    coordinate = min(max((confidence - start) / (full - start), 0.0), 1.0)
    smooth_confidence = coordinate * coordinate * (3.0 - 2.0 * coordinate)
    return 1.0 - smooth_confidence, confidence


def load_correspondence(path, latent_hw, device):
    """Load the public NPZ map contract and project it to the VAE lattice."""

    required = (
        "source_to_target",
        "target_to_source",
        "source_reliability",
        "target_reliability",
    )
    with np.load(path) as payload:
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        correspondence = Correspondence(
            *(torch.from_numpy(payload[key]).float().to(device) for key in required)
        )
    return correspondence.to_grids(tuple(latent_hw))
