# Model setup

AlignMorph and the bundled FreeMorph runner use the same Stable Diffusion 2.1
checkpoint. Run these commands from the repository root in the `alignmorph`
environment.

## Download

```bash
python scripts/prepare_model.py --output models/sd21_768
```

The helper downloads the weights, copies the bundled configuration files, and
checks every file against [the model manifest](../configs/model.json).
To verify an existing installation without downloading files:

```bash
python scripts/prepare_model.py --output models/sd21_768 --verify-only
```

## Use local files

If you already have the matching checkpoint files in the component layout
listed in the manifest, copy and verify them with:

```bash
python scripts/prepare_model.py --from-local /path/to/matching/checkpoint \
  --output models/sd21_768
```

The source directory must contain the exact files listed in the manifest.
A different checkpoint will fail the hash check. If an output directory contains
mismatched files, use a new directory.

## Checkpoint layout

The manifest pins repository revisions and SHA-256 hashes. Small configuration
files are bundled in `configs/checkpoint_layout`. UNet weights come from the
SD2.1-768 mirror; the text encoder and VAE come from the SD2.1-base mirror, with
matching component configurations. UNet files are fp16, while text encoder and
VAE files are full precision. At inference, the pipeline loads the UNet and VAE
as fp16 and keeps the text encoder in full precision.

Model weights are not bundled and retain their upstream license.

## Optional captioning

AlignMorph reads the two endpoint captions from the `prompts` field in the pair
manifest. It does not load a captioning model during generation. All 21 bundled
pairs already include captions; custom captions can also be written manually.

For automatic captions, follow [FreeMorph's captioning workflow](https://github.com/yukangcao/FreeMorph#captioning-the-image-pairs).
Its [caption script](https://github.com/yukangcao/FreeMorph/blob/b7b7068291c6654232f8c8959e22214e311ede69/caption.py)
uses LLaVA-NeXT / LLaVA 1.6 with the checkpoint
[`llava-hf/llava-v1.6-mistral-7b-hf`](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf).
Follow that model's installation and usage instructions in a separate captioning
environment with support for `LlavaNextProcessor` and
`LlavaNextForConditionalGeneration`. The generation environment's pinned
`transformers==4.34.1` does not provide these classes; keep its dependencies
unchanged when setting up captioning.

Caption generation is an optional upstream workflow; the captioning script and
its environment are not bundled here. Save the resulting source and target
captions in `prompts`, then return to the `alignmorph` environment to run image
morphing. See [the pair format](../README.md#custom-image-pairs).
