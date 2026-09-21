# AlignMorph

Official implementation of **AlignMorph**.

AlignMorph is a training-free method for image morphing with semantic
correspondence and multiband latent transport. Given two images and their
captions, it generates a seven-frame sequence connecting the endpoints.

## Example comparisons

The examples show morphing between semantically corresponding subjects with
large spatial displacements. Each figure shows **FreeMorph on top** and
**AlignMorph below**.

### car_01

![car_01: FreeMorph and AlignMorph](examples/results/comparison/car_01.jpg)

### dog_01

![dog_01: FreeMorph and AlignMorph](examples/results/comparison/dog_01.jpg)

### dog_08

![dog_08: FreeMorph and AlignMorph](examples/results/comparison/dog_08.jpg)

See the [example guide](examples/README.md) to regenerate both methods and the
comparison figures, or browse the [gallery](examples/index.html).

## Installation

Clone the repository and run the following commands from its root directory:

```bash
git clone https://github.com/51xOne/Alignmorph.git
cd Alignmorph
conda create -n alignmorph python=3.9 -y
conda activate alignmorph
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python scripts/prepare_model.py --output models/sd21_768
```

Generation requires a CUDA-capable GPU. The examples were run with Python 3.9,
PyTorch 2.8.0, CUDA 12.8, and a GPU with 24 GB memory. The
[requirements file](requirements.txt) pins the generation dependencies, including
`diffusers==0.17.1` and `transformers==4.34.1`. The model preparation command
downloads and verifies the Stable Diffusion 2.1 checkpoint; see
[model setup](docs/models.md) for using matching local files.

Captions are already provided for all 21 pairs. LLaVA is optional preprocessing
for new images and is not required to run these examples.

## Quick start

Correspondence maps are bundled for `car_01`, `dog_01`, and `dog_08`. Generate one
example with:

```bash
python run.py --pairs examples/pairs21.jsonl \
  --correspondence examples/correspondence --model models/sd21_768 \
  --ids car_01 --output outputs/example
```

Use `--ids car_01 dog_01 dog_08` to generate all three. Each pair produces a
seven-frame PNG strip and JSON run records. Choose a new output directory for
each run; existing results are not overwritten. Keep the full manifest and use
`--ids` to select examples while preserving the sampling order.

## The 21 image pairs

We provide 21 pairs of cars, motorcycles, and dogs with semantic correspondence
and large spatial displacements. Corresponding parts occupy different positions
across the endpoints, with changes in pose and layout, making these pairs useful
for examining morphing under substantial spatial misalignment.

All images and captions are included. The [full manifest](examples/pairs21.jsonl)
records each pair's images, captions, and ID; the [pair index](examples/images/README.md)
lists all source and target images. The selection combines new endpoint pairings
with original dataset pairs. Photographs come from Morph4Data; see
[attribution](NOTICE.md). Shared endpoints are stored once in `examples/images`.

Only the three examples above include precomputed correspondence maps. To run
all 21 pairs, first follow the [feature environment setup](docs/correspondence.md#feature-environment),
then generate the maps:

```bash
conda activate alignmorph-features
python scripts/generate_correspondence.py --json_path examples/pairs21.jsonl \
  --output_dir outputs/correspondence21 \
  --featup_repo third_party/FeatUp --dinov2_repo third_party/dinov2
```

Switch back to the generation environment:

```bash
conda activate alignmorph
python run.py --pairs examples/pairs21.jsonl \
  --correspondence outputs/correspondence21 --model models/sd21_768 \
  --output outputs/alignmorph21
```

## Custom image pairs

Provide one caption for each endpoint, written manually or generated with an
image-captioning model. For automatic captions, follow the
[optional captioning guide](docs/models.md#optional-captioning). Create a JSONL
file with one pair per line; image paths are relative to that file:

```json
{"exp_id": "car", "image_paths": ["images/source.png", "images/target.png"], "prompts": ["a red sports car", "a blue sports car"]}
```

[Extract correspondence maps](docs/correspondence.md#custom-pairs), then generate:

```bash
conda activate alignmorph
python run.py --pairs pairs.jsonl --correspondence outputs/correspondence \
  --model models/sd21_768 --output outputs/morphs
```

## Warping visualization

Visualize latent transport and its low- and high-frequency components:

```bash
python scripts/visualize_warping.py --model models/sd21_768 \
  --ids car_01 dog_01 dog_08 --output outputs/warping
```

Open `outputs/warping/index.html` to inspect the full latent warp, multiband
transport, frequency components, and draft path. Raw tensors are also saved.
This command uses the VAE without running diffusion denoising.

See the [visualization guide](docs/warping.md) for interpreting the figures and
visualizing the exact sampled latents from a generation run.

## Acknowledgements

Built on [FreeMorph](https://github.com/yukangcao/FreeMorph),
[FeatUp](https://github.com/mhamilton723/FeatUp),
[DINOv2](https://github.com/facebookresearch/dinov2), and Stable Diffusion.
See [NOTICE.md](NOTICE.md) for attribution.
