"""Feature extraction settings used by the released correspondence maps."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrespondenceConfig:
    image_size: int = 504
    max_tokens: int = 8192
    projection_mode: str = "local_soft"
    feature_recipe: str = "normalized_model_grid"
