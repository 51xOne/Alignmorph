"""Visualize latent, low/high-band transport, and the bi-phase draft operands."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignmorph.io import read_pairs
from alignmorph.sequence import select_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=ROOT / "examples/pairs.jsonl")
    p.add_argument(
        "--correspondence", type=Path, default=ROOT / "examples/correspondence"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--latents-dir",
        type=Path,
        help="Use exact endpoints from run.py --save-latents; otherwise use VAE posterior means",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Endpoint transport display strength only; draft retains the released midpoint setting",
    )
    a = p.parse_args()
    pairs = select_pairs(read_pairs(a.pairs), a.ids)
    for pair in pairs:
        name = pair["exp_id"]
        if (a.output / name).exists():
            p.error(f"Output already exists: {name}")
        if not (a.correspondence / f"{name}.npz").is_file():
            p.error(f"Missing correspondence: {name}")
        if a.latents_dir and not (a.latents_dir / f"{name}.pt").is_file():
            p.error(f"Missing saved latents: {name}")
    from alignmorph.visualization import WarpVisualizer

    visualizer = WarpVisualizer(a.model, a.device)
    for pair in pairs:
        name = pair["exp_id"]
        saved = a.latents_dir / f"{name}.pt" if a.latents_dir else None
        visualizer.render(
            pair,
            a.correspondence / f"{name}.npz",
            a.output,
            latents_path=saved,
            strength=a.strength,
        )
        print(f'Saved {a.output/name/"index.html"}', flush=True)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><h1>AlignMorph warping</h1>'
        + "".join(
            f'<p><a href="{x["exp_id"]}/index.html">{x["exp_id"]}</a></p>'
            for x in pairs
        )
    )


if __name__ == "__main__":
    main()
