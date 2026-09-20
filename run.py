"""Generate seven-frame morphs using the released sampling configuration."""

import argparse
import os
from pathlib import Path

from alignmorph.io import check_outputs, read_pairs, record_run
from alignmorph.sequence import generate_sequence, select_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--correspondence", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Local SD 2.1-768 checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--ids", nargs="+", help="Select IDs while preserving manifest RNG order"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check", action="store_true", help="Check inputs without loading a model"
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save exact sampled endpoints and draft for visualization",
    )
    args = parser.parse_args()
    pairs = read_pairs(args.pairs, args.image_root)
    selected = select_pairs(pairs, args.ids)
    check_outputs(selected, args.output, args.correspondence)
    if args.check:
        print(f"{len(selected)} pairs ready; sampling order comes from {args.pairs}")
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from alignmorph.pipeline import AlignMorphPipeline

    pipeline = AlignMorphPipeline(args.model, args.device, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    def generate(pair):
        name = pair["exp_id"]
        print(f"Generating {name}", flush=True)
        correspondence = args.correspondence / f"{name}.npz"
        output = args.output / f"{name}.png"
        pipeline.generate(pair, correspondence, output, save_latents=args.save_latents)
        record_run(
            pair,
            output,
            correspondence,
            dict(
                method="AlignMorph",
                seed=args.seed,
                model=args.model,
                frames=7,
                steps=50,
                edit_strength=0.8,
                guidance_scale=7.5,
                aligned_reference_max_weight=0.25,
                confidence_thresholds=[0.1, 0.2],
                pair_order=[p["exp_id"] for p in pairs],
            ),
        )

    generate_sequence(pipeline, pairs, selected, generate)


if __name__ == "__main__":
    main()
