"""Run the unchanged official FreeMorph entrypoint with local model files."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignmorph.io import check_outputs, read_pairs, sha256


def gpu_environment(device):
    env = dict(os.environ)
    if device == "cuda":
        return env
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("FreeMorph requires a CUDA device")
    index = int(device[5:])
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else None
    env["CUDA_VISIBLE_DEVICES"] = devices[index] if devices else str(index)
    return env


def run(pairs_path, model, output, device="cuda"):
    pairs = read_pairs(pairs_path)
    output = Path(output).resolve()
    check_outputs(pairs, output)
    source = ROOT / "third_party/freemorph"
    manifest = json.loads((source / "SOURCE.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(source / name) != expected:
            raise ValueError(f"Official FreeMorph source differs: {name}")
    model = Path(model).resolve()
    if not model.is_dir():
        raise FileNotFoundError(model)
    env = gpu_environment(device)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="freemorph-") as folder:
        work = Path(folder)
        (work / "stabilityai").mkdir()
        (work / "stabilityai/stable-diffusion-2-1").symlink_to(
            model, target_is_directory=True
        )
        (work / "eval_results").mkdir()
        (work / "eval_results/freemorph").symlink_to(output, target_is_directory=True)
        inputs = work / "pairs.jsonl"
        inputs.write_text("".join(json.dumps(pair) + "\n" for pair in pairs))
        command = [
            sys.executable,
            str(source / "freemorph.py"),
            "--json_path",
            str(inputs),
        ]
        subprocess.run(command, cwd=work, env=env, check=True)
    record = dict(
        revision=manifest["revision"],
        source_modified=False,
        model=str(model),
        pair_order=[p["exp_id"] for p in pairs],
        outputs={
            f'{p["exp_id"]}.png': sha256(output / f'{p["exp_id"]}.png') for p in pairs
        },
    )
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.pairs, args.model, args.output, args.device)


if __name__ == "__main__":
    main()
