# Examples

The repository includes 21 pairs of semantically corresponding cars, motorcycles,
and dogs with large spatial displacements. All input images and captions are
included in [pairs21.jsonl](pairs21.jsonl); see the [pair index](images/README.md).
Precomputed maps and comparison figures are provided for `car_01`, `dog_01`, and
`dog_08`. Follow the [correspondence guide](../docs/correspondence.md#all-21-pairs)
to extract maps for the complete set.

[Open the comparison gallery](index.html). Each figure shows **FreeMorph above
AlignMorph**. The included figures are compact JPEGs; the commands below produce
full-resolution PNG results.

## Generate AlignMorph examples

Complete the [installation](../README.md#installation), then run from the
repository root in the `alignmorph` environment:

```bash
python run.py --pairs examples/pairs21.jsonl \
  --correspondence examples/correspondence --model models/sd21_768 \
  --ids car_01 dog_01 dog_08 --output outputs/alignmorph
```

For a single pair, replace the IDs with `--ids car_01`. Keep `pairs21.jsonl` as the
input manifest. No correspondence maps are needed for skipped pairs.

## Regenerate comparisons with FreeMorph

The bundled runner uses the pinned [official FreeMorph source](../third_party/freemorph/SOURCE.json).
Generate its results using the full manifest, then select the three examples
when building the comparison figures:

```bash
python scripts/run_freemorph.py --pairs examples/pairs21.jsonl \
  --model models/sd21_768 --output outputs/freemorph
python scripts/compare.py --pairs examples/pairs.jsonl \
  --freemorph outputs/freemorph --alignmorph outputs/alignmorph \
  --output outputs/comparison
```

FreeMorph generates all 21 pairs in this command. The comparison script uses the
three-pair manifest to select the illustrated examples. Each comparison is saved
as a full-resolution PNG and a smaller JPEG. Use new output directories when
rerunning commands.

## Sampling and outputs

The examples use seed 42, 768-pixel images, seven frames, and guidance scale 7.5.
Both methods use the same checkpoint, with 50 scheduler steps and 40 active
inversion and denoising steps. Seed 42 is set once at startup, following the
baseline's sequential sampling. Changing the manifest changes subsequent VAE
samples, so the commands above retain the full 21-pair order. CUDA numerics may
vary across runs and hardware.

See the [visualization guide](../docs/warping.md) to inspect latent transport or
load the exact sampled latents from a generation run.
