#!/usr/bin/env python
"""Generate the public AlignMorph NPZ correspondence contract from JSONL pairs."""

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignmorph.correspondence import dense_ot_correspondence
from alignmorph.config import CorrespondenceConfig
from alignmorph.io import read_pairs


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--featup_repo",
        default="mhamilton723/FeatUp",
        help="Local FeatUp checkout or Torch Hub repository.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dinov2_repo",
        default=None,
        help="Optional local DINOv2 checkout for FeatUp's nested backbone load.",
    )
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--base_downsample", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument(
        "--projection_mode", choices=("soft", "local_soft"), default=None
    )
    parser.add_argument("--local_soft_radius", type=float, default=4.0)
    parser.add_argument(
        "--feature_recipe", choices=("reference", "normalized_model_grid"), default=None
    )
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn_iterations", type=int, default=100)
    parser.add_argument("--cycle_threshold_px", type=float, default=8.0)
    parser.add_argument("--cycle_sigma_px", type=float, default=4.0)
    parser.add_argument("--minimum_confidence", type=float, default=0.2)
    parser.add_argument(
        "--exp_ids", default=None, help="Optional comma-separated subset."
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.preset = "alignmorph"
    config = CorrespondenceConfig()
    args.image_size = (
        args.image_size if args.image_size is not None else config.image_size
    )
    args.max_tokens = (
        args.max_tokens if args.max_tokens is not None else config.max_tokens
    )
    args.projection_mode = args.projection_mode or config.projection_mode
    args.feature_recipe = args.feature_recipe or config.feature_recipe
    return args


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_provenance(model):
    """Bind correspondence caches to loaded weights and feature implementation."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    sources = {}
    for module in model.modules():
        cls = type(module)
        if cls.__module__.startswith(("featup", "dinov2")) or cls is type(model):
            try:
                path = inspect.getsourcefile(cls)
            except (TypeError, OSError):
                # Torch Hub executes hubconf without registering its class
                # module; the Python forward function retains its source file.
                path = inspect.getsourcefile(module.forward)
            if path:
                sources[cls.__module__] = sha256(path)
    extensions = {}
    for name in ("adaptive_conv_cuda_impl", "adaptive_conv_cpp_impl"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            extensions[name] = sha256(module.__file__)
    return dict(
        state_dict_sha256=digest.hexdigest(),
        feature_source_sha256=sources,
        extension_sha256=extensions,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def load_image(path, image_size, device, floating_crop=False):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) / 2 if floating_crop else (width - edge) // 2
    top = (height - edge) / 2 if floating_crop else (height - edge) // 2
    image = image.crop((left, top, left + edge, top + edge))
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    return tensor.unsqueeze(0).to(device)


def effective_downsample(height, width, base, max_tokens):
    required = int(math.ceil(math.sqrt((height * width) / float(max_tokens))))
    return max(int(base), required)


def extract_features(model, image, max_tokens, base_downsample, stable_recipe=False):
    with torch.no_grad():
        features = model(image)
    if features.shape[-2:] != image.shape[-2:]:
        features = F.interpolate(
            features, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
    if stable_recipe:
        features = F.normalize(
            features[0].permute(1, 2, 0).contiguous(), p=2, dim=-1
        ).permute(2, 0, 1)[None]
    downsample = effective_downsample(
        features.shape[-2], features.shape[-1], base_downsample, max_tokens
    )
    output_hw = (
        max(1, features.shape[-2] // downsample),
        max(1, features.shape[-1] // downsample),
    )
    if stable_recipe:
        features = F.interpolate(
            features,
            scale_factor=1.0 / downsample,
            mode="bilinear",
            align_corners=False,
        )
        return features[0], downsample
    features = F.interpolate(
        features, size=output_hw, mode="bilinear", align_corners=False
    )
    return F.normalize(features[0].float(), p=2, dim=0), downsample


@contextmanager
def pinned_dinov2_loader(repo):
    """Route FeatUp's nested Hub request to a specified local backbone checkout.

    Used only during single-threaded model initialization; always restores Hub.
    """
    if repo is None:
        yield
        return
    directory = Path(repo).resolve()
    if not (directory / "hubconf.py").is_file():
        raise ValueError("dinov2_repo must contain a local hubconf.py")
    original = torch.hub.load
    routed = []

    def load(repository, model, *args, **kwargs):
        if str(repository).split(":", 1)[0] == "facebookresearch/dinov2":
            routed.append(model)
            kwargs["source"] = "local"
            return original(str(directory), model, *args, **kwargs)
        return original(repository, model, *args, **kwargs)

    torch.hub.load = load
    try:
        yield
        if not routed:
            raise RuntimeError("FeatUp did not request the pinned DINOv2 backbone")
    finally:
        torch.hub.load = original


def load_featup(repo, device, dinov2_repo=None):
    local = Path(repo).is_dir()
    with pinned_dinov2_loader(dinov2_repo):
        model = torch.hub.load(
            os.fspath(Path(repo).resolve()) if local else repo,
            "dinov2",
            source="local" if local else "github",
            pretrained=True,
            use_norm=True,
            trust_repo=True,
        )
    return model.to(device).eval()


def main():
    args = parse_args()
    stable_recipe = args.feature_recipe == "normalized_model_grid"
    if args.image_size <= 0 or args.image_size % 14 != 0:
        raise ValueError(
            "image_size must be a positive multiple of DINOv2's 14px patch"
        )
    if args.base_downsample <= 0 or args.max_tokens <= 0:
        raise ValueError("downsample and max_tokens must be positive")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = read_pairs(args.json_path)
    ids = [str(entry["exp_id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate exp_id in input JSONL")
    if any(
        not value or Path(value).name != value or value in (".", "..") for value in ids
    ):
        raise ValueError("exp_id must be a single nonempty filename component")
    if args.exp_ids is not None:
        selected = {item.strip() for item in args.exp_ids.split(",") if item.strip()}
        if selected - set(ids):
            raise ValueError(
                f"Requested exp_ids absent from JSONL: {sorted(selected - set(ids))}"
            )
        entries = [entry for entry in entries if str(entry["exp_id"]) in selected]
    recipe = {
        key: value
        for key, value in vars(args).items()
        if key not in ("output_dir", "json_path", "exp_ids", "overwrite", "device")
    }
    recipe["source_sha256"] = {
        name: sha256(REPO_ROOT / name)
        for name in (
            "scripts/generate_correspondence.py",
            "alignmorph/correspondence.py",
            "alignmorph/geometry.py",
            "alignmorph/config.py",
        )
    }
    torch.manual_seed(0)
    model = load_featup(args.featup_repo, device, args.dinov2_repo)
    recipe["model"] = model_provenance(model)
    recipe_fingerprint = hashlib.sha256(
        json.dumps(recipe, sort_keys=True).encode()
    ).hexdigest()

    for entry in tqdm(entries):
        exp_id = str(entry["exp_id"])
        output_path = output_dir / f"{exp_id}.npz"
        metadata_path = output_dir / f"{exp_id}.json"
        source_path, target_path = entry["image_paths"]
        source_hash, target_hash = sha256(source_path), sha256(target_path)
        if output_path.exists() or metadata_path.exists():
            if not args.overwrite:
                existing = (
                    json.loads(metadata_path.read_text())
                    if metadata_path.exists()
                    else {}
                )
                if (
                    output_path.exists()
                    and existing.get("recipe_fingerprint") == recipe_fingerprint
                    and existing.get("source_sha256") == source_hash
                    and existing.get("target_sha256") == target_hash
                    and existing.get("npz_sha256") == sha256(output_path)
                ):
                    continue
                raise FileExistsError(
                    f"Existing correspondence has different or unverified provenance: {output_path}; use a new output directory or --overwrite"
                )
        source_image = load_image(
            source_path, args.image_size, device, floating_crop=stable_recipe
        )
        target_image = load_image(
            target_path, args.image_size, device, floating_crop=stable_recipe
        )
        source_features, source_downsample = extract_features(
            model,
            source_image,
            args.max_tokens,
            args.base_downsample,
            stable_recipe=stable_recipe,
        )
        target_features, target_downsample = extract_features(
            model,
            target_image,
            args.max_tokens,
            args.base_downsample,
            stable_recipe=stable_recipe,
        )
        correspondence = dense_ot_correspondence(
            source_features,
            target_features,
            epsilon=args.sinkhorn_epsilon,
            iterations=args.sinkhorn_iterations,
            max_tokens=args.max_tokens,
            cycle_threshold_px=args.cycle_threshold_px,
            cycle_sigma_px=args.cycle_sigma_px,
            minimum_confidence=args.minimum_confidence,
            projection_mode=args.projection_mode,
            local_soft_radius=args.local_soft_radius,
            independent_reverse=stable_recipe,
            normalization_epsilon=1e-8 if stable_recipe else 0.0,
            reliability_on_cpu=stable_recipe,
        )
        ot_grid = list(correspondence.source_to_target.shape[:2])
        if stable_recipe:
            correspondence = correspondence.to_grids(tuple(source_image.shape[-2:]))
        np.savez_compressed(
            output_path,
            source_to_target=correspondence.source_to_target.cpu().numpy(),
            target_to_source=correspondence.target_to_source.cpu().numpy(),
            source_reliability=correspondence.source_reliability.cpu().numpy(),
            target_reliability=correspondence.target_reliability.cpu().numpy(),
        )
        metadata = {
            "exp_id": entry["exp_id"],
            "preset": args.preset,
            "feature_recipe": args.feature_recipe,
            "projection_mode": args.projection_mode,
            "local_soft_radius": args.local_soft_radius,
            "ot_grid": ot_grid,
            "independent_reverse": stable_recipe,
            "normalization_epsilon": 1e-8 if stable_recipe else 0.0,
            "normalize_before_downsample": stable_recipe,
            "reliability_device": "cpu" if stable_recipe else str(device),
            "source_path": source_path,
            "target_path": target_path,
            "source_sha256": source_hash,
            "target_sha256": target_hash,
            "recipe": recipe,
            "recipe_fingerprint": recipe_fingerprint,
            "npz_sha256": sha256(output_path),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "feature_grid_source": list(correspondence.source_to_target.shape[:2]),
            "feature_grid_target": list(correspondence.target_to_source.shape[:2]),
            "source_downsample": source_downsample,
            "target_downsample": target_downsample,
            "featup_repo": args.featup_repo,
            "backbone": "dinov2_vits14_featup",
            "image_size": args.image_size,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "max_tokens": args.max_tokens,
            "cycle_threshold_px": args.cycle_threshold_px,
            "cycle_sigma_px": args.cycle_sigma_px,
            "minimum_confidence": args.minimum_confidence,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
