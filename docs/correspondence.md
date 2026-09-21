# Correspondence extraction

The three runnable examples already include their correspondence maps. Follow
this guide to extract maps for all 21 pairs or for your own images. Run commands
from the repository root.

## Feature environment

Use a separate environment for FeatUp. Its CUDA extension requires CUDA Toolkit
12.4, including `nvcc`, and a C++17 compiler. The CUDA runtime installed by pip
does not include the compiler. Set `CUDA_HOME` below to your toolkit location.

```bash
conda create -n alignmorph-features python=3.10 -y
conda activate alignmorph-features
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-features.txt
git clone https://github.com/mhamilton723/FeatUp.git third_party/FeatUp
git -C third_party/FeatUp checkout --detach 6b5a6c0e91f75e69194807128dcbc39c3084a30d
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
cd third_party/FeatUp
CUDA_HOME=/usr/local/cuda-12.4 MAX_JOBS=2 python setup.py build_ext --inplace
cd ../..
```

## All 21 pairs

In the feature environment, generate maps into a new output directory:

```bash
python scripts/generate_correspondence.py --json_path examples/pairs21.jsonl \
  --output_dir outputs/correspondence21 \
  --featup_repo third_party/FeatUp --dinov2_repo third_party/dinov2
```

The manifest includes all 21 pairs' image paths and captions. Only three maps
are bundled with the repository; this command generates the complete set.

## Custom pairs

Create `pairs.jsonl` as described in the [README](../README.md#custom-image-pairs),
then run:

```bash
python scripts/generate_correspondence.py --json_path pairs.jsonl \
  --output_dir outputs/correspondence \
  --featup_repo third_party/FeatUp --dinov2_repo third_party/dinov2
```

Each pair produces an NPZ with both directional maps and correspondence
reliability, plus a JSON record of settings and input hashes. Defaults use
504-pixel feature inputs, at most 8192 tokens, and local OT barycentric
projection with radius 4.

## Generate images

Switch back to the generation environment and pass the same pair manifest and
the new correspondence directory:

```bash
conda activate alignmorph
python run.py --pairs pairs.jsonl --correspondence outputs/correspondence \
  --model models/sd21_768 --output outputs/morphs
```

For all 21 pairs, use `--pairs examples/pairs21.jsonl` and
`--correspondence outputs/correspondence21`. The same two arguments work with
`scripts/visualize_warping.py`.
