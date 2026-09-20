# Examples

Three runnable examples are included: `car_01`, `dog_01`, and `dog_08`.
Their input images, captions and correspondence maps are bundled. All 21 pairs'
images and captions are also included in [pairs21.jsonl](pairs21.jsonl); see the
[pair index](images/README.md). The other 18 correspondence maps can be generated
with the [feature extraction command](../docs/correspondence.md#all-21-pairs).

[Open the comparison gallery](index.html). FreeMorph is the first row and
AlignMorph is the second. The package includes compact JPEG comparison figures.
Use the commands below to generate full-resolution PNG results.

## Generate the examples

Run these commands from the repository root:

```bash
python scripts/prepare_model.py --output models/sd21_768 --verify-only
python run.py --pairs examples/pairs21.jsonl \
  --correspondence examples/correspondence --model models/sd21_768 \
  --ids car_01 dog_01 dog_08 --output outputs/alignmorph
python scripts/run_freemorph.py --pairs examples/pairs21.jsonl \
  --model models/sd21_768 --output outputs/freemorph
python scripts/compare.py --pairs examples/pairs.jsonl --freemorph outputs/freemorph \
  --alignmorph outputs/alignmorph --output outputs/comparison
```

Use `run.py --ids car_01` for a single AlignMorph example. The full manifest order
is preserved when replaying the VAE sampling prefix. The examples use seed 42,
768-pixel images, seven frames, 50 scheduler steps with 40 active inversion and
denoising steps, and guidance scale 7.5. Both methods use the same checkpoint.
Saved figures use the full 21-pair sampling order. The commands above retain
that order: AlignMorph generates the three selected examples; FreeMorph generates
all 21 pairs, and the comparison command selects the three illustrated pairs.
Seed 42 is set once at startup, following the baseline's sequential sampling.
Changing the input manifest changes subsequent VAE samples. CUDA backend numerics
can also differ across runs or hardware.

Warping visualization is described in [the visualization guide](../docs/warping.md).
The saved results are illustrative examples, not a general-purpose benchmark.

## Browse the comparisons

### car_01

![car_01](results/comparison/car_01.jpg)

### dog_01

![dog_01](results/comparison/dog_01.jpg)

### dog_08

![dog_08](results/comparison/dog_08.jpg)
