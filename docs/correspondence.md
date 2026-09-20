# Correspondence extraction

The three runnable examples include their correspondence maps. For new images, use a
separate feature environment, since FeatUp's CUDA extension must match the
installed PyTorch and CUDA toolkit. This build requires CUDA Toolkit 12.4
(including `nvcc`) and a C++17 compiler; the CUDA runtime installed by pip alone
does not provide the compiler.

```bash
conda create -n alignmorph-features python=3.10 -y
conda activate alignmorph-features
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-features.txt
git clone https://github.com/mhamilton723/FeatUp.git third_party/FeatUp
git -C third_party/FeatUp checkout --detach 6b5a6c0e91f75e69194807128dcbc39c3084a30d
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8
cd third_party/FeatUp
CUDA_HOME=/usr/local/cuda-12.4 MAX_JOBS=2 python setup.py build_ext --inplace
cd ../..
python scripts/generate_correspondence.py \
  --json_path pairs.jsonl \
  --output_dir outputs/correspondence \
  --featup_repo third_party/FeatUp \
  --dinov2_repo third_party/dinov2
```

Set `CUDA_HOME` to the matching toolkit on your machine. Defaults are 504-pixel
feature inputs, at most 8192 tokens, and local OT barycentric projection with
radius 4. Both directional maps and cycle-based reliability fields are saved in
each NPZ. A JSON sidecar records the recipe and input hashes.

## All 21 pairs

All input images and captions are bundled in `examples/images` and
`examples/pairs21.jsonl`. Only three precomputed maps are included to keep the
package small. In the feature environment, generate the full set into a new
output directory:

```bash
python scripts/generate_correspondence.py --json_path examples/pairs21.jsonl \
  --output_dir outputs/correspondence21 \
  --featup_repo third_party/FeatUp --dinov2_repo third_party/dinov2
```

Then use `--pairs examples/pairs21.jsonl --correspondence outputs/correspondence21`
with `run.py` in the generation environment. The same arguments select the full
set in `scripts/visualize_warping.py`.
