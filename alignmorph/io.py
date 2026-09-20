"""Pair manifests and reproducible run records."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pairs(path, image_root=None):
    path = Path(path).resolve()
    root = Path(image_root).resolve() if image_root else path.parent
    pairs, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        name = str(pair["exp_id"])
        if not name or name in (".", "..") or any(c in name for c in "/\\\x00"):
            raise ValueError(f"Invalid pair ID: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate pair ID: {name}")
        seen.add(name)
        for field in ("image_paths", "prompts"):
            if (
                not isinstance(pair.get(field), list)
                or len(pair[field]) != 2
                or not all(isinstance(x, str) for x in pair[field])
            ):
                raise ValueError(f"{name}: {field} must contain two strings")
        pair = dict(
            pair,
            exp_id=name,
            image_paths=[str((root / p).resolve()) for p in pair["image_paths"]],
        )
        for image in pair["image_paths"]:
            if not Path(image).is_file():
                raise FileNotFoundError(image)
        pairs.append(pair)
    if not pairs:
        raise ValueError("The pair manifest is empty")
    return pairs


def check_outputs(pairs, output, correspondence=None):
    output = Path(output)
    for pair in pairs:
        name = pair["exp_id"]
        for suffix in (".png", ".json", ".run.json"):
            if (output / (name + suffix)).exists():
                raise FileExistsError(
                    f"{name} already has output; choose a new output directory"
                )
        if correspondence and not (Path(correspondence) / (name + ".npz")).is_file():
            raise FileNotFoundError(f"Missing correspondence: {name}.npz")


def record_run(pair, output, correspondence, settings):
    output = Path(output)
    record = dict(
        pair=pair["exp_id"],
        prompts=pair["prompts"],
        settings=settings,
        inputs=[dict(name=Path(p).name, sha256=sha256(p)) for p in pair["image_paths"]],
        correspondence_sha256=sha256(correspondence),
        output_sha256=sha256(output),
    )
    output.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
