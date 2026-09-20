"""Download or verify the checkpoint files used by AlignMorph."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignmorph.io import sha256 as file_sha256


def prepare(output, manifest, *, source=None, verify_only=False):
    output = Path(output)
    paths = set(manifest["files"]) | set(manifest["aliases"])
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in paths):
        raise ValueError("Model manifest paths must stay inside the output directory")
    if not set(manifest["aliases"].values()) <= set(manifest["files"]):
        raise ValueError("Alias targets must be declared model files")
    for name, expected in manifest["files"].items():
        path = output / name
        if path.exists():
            if file_sha256(path) != expected:
                raise ValueError(
                    f"Existing model file differs: {path}; use a new directory"
                )
        elif verify_only:
            raise FileNotFoundError(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if source is not None:
                origin = Path(source) / name
                if file_sha256(origin) != expected:
                    raise ValueError(f"Source model file differs: {origin}")
                shutil.copy2(origin, path)
            else:
                origin = manifest.get("sources", {}).get(name, manifest)
                if "bundled" in origin:
                    shutil.copy2(ROOT / origin["bundled"], path)
                else:
                    from huggingface_hub import hf_hub_download

                    hf_hub_download(
                        origin["repo_id"],
                        filename=name,
                        revision=origin["revision"],
                        local_dir=str(output),
                        local_dir_use_symlinks=False,
                    )
            if file_sha256(path) != expected:
                raise ValueError(f"Downloaded/copied model checksum mismatch: {path}")
    for alias, target in manifest["aliases"].items():
        path = output / alias
        if path.exists() or path.is_symlink():
            if not path.is_file() or file_sha256(path) != manifest["files"][target]:
                raise ValueError(f"Existing model alias differs: {path}")
        elif verify_only:
            raise FileNotFoundError(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(output / target, path)
            except OSError:
                shutil.copy2(output / target, path)
    if not verify_only:
        (output / "alignmorph_model_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--from-local",
        type=Path,
        default=None,
        help="Copy a locally available matching checkpoint instead of downloading it.",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/model.json").read_text())
    print(
        prepare(
            args.output, manifest, source=args.from_local, verify_only=args.verify_only
        )
    )
