"""Make two-row FreeMorph / AlignMorph figures from generated strips."""

import argparse, json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alignmorph.io import read_pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, default=ROOT / "examples/pairs.jsonl")
    p.add_argument(
        "--freemorph", type=Path, required=True
    )
    p.add_argument(
        "--alignmorph", type=Path, required=True
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    pairs = read_pairs(a.pairs)
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        f = ImageFont.load_default()
    for pair in pairs:
        uid = pair["exp_id"]
        output = a.output / f"{uid}.png"
        if output.exists():
            raise FileExistsError(output)
        images = [
            Image.open(folder / f"{uid}.png").convert("RGB")
            for folder in [a.freemorph, a.alignmorph]
        ]
        if images[0].size != images[1].size:
            raise ValueError(f"{uid}: strip dimensions differ")
        w, h = images[0].size
        canvas = Image.new("RGB", (w, 2 * (h + 48)), "white")
        d = ImageDraw.Draw(canvas)
        for j, (im, label) in enumerate(zip(images, ["FreeMorph", "AlignMorph"])):
            y = j * (h + 48)
            d.text((12, y + 7), label, font=f, fill="black")
            canvas.paste(im, (0, y + 48))
        canvas.save(output)
        canvas.resize(
            (1800, round(canvas.height * 1800 / w)), Image.Resampling.LANCZOS
        ).save(output.with_suffix(".jpg"), quality=92)
        print(output, flush=True)


if __name__ == "__main__":
    main()
