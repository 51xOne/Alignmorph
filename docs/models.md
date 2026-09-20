# Model setup

Both methods use the model files listed in
`configs/model.json`. 

```bash
python scripts/prepare_model.py --output models/sd21_768
python scripts/prepare_model.py --output models/sd21_768 --verify-only
```

The UNet files are fp16; the text encoder and VAE files are full precision.
The pipeline loads the VAE and UNet as fp16 and keeps the text encoder in full
precision. The manifest pins each file's repository revision and SHA256 hash;
small configuration files are bundled in `configs/checkpoint_layout`.
The UNet comes from the SD2.1-768 mirror and the text encoder and VAE from the
SD2.1-base mirror, with matching component configurations.

`--from-local /path/to/matching/files` prepares the same layout from a local
checkpoint. Use `--verify-only` to check the checkpoint before generation. Model files retain their upstream license and are not bundled.
