# Warping visualization

`python scripts/visualize_warping.py --model models/sd21_768 --ids car_01 --output outputs/warping`

Each pair produces an HTML page, seven figures, `metadata.json`, and `tensors.pt`.
The figures show transport before diffusion refinement.

| Figure | Contents |
| --- | --- |
| `latent.png` | VAE endpoint reconstructions, full latent warp `W(z)`, and multiband transport |
| `low_frequency.png` | Decoded original low-frequency latent `L` and warped low-frequency latent `W(L)` |
| `high_frequency.png` | Four-channel signed plots of `H`, `W(H)`, `w W(H)`, and `(1-w) H` |
| `path.png` | Each frame's draft and its two actual Slerp operands |
| `midpoint.png` | Source-chart blend, target-chart blend, transported target blend, and final midpoint |
| `reliability.png` | The reliability used for each direction; black is zero, white is one |
| `confidence_gates.png` | Bidirectional confidence and the spatial gates for aligned K/V |

The transport operator is

```text
L = average_pool(z, kernel=9, replicate padding)
H = z - L
transport(z) = W(L) + w * W(H) + (1 - w) * H
```

Low frequency is resampled spatially. High frequency is a mixture of transported
and retained content, controlled by correspondence reliability. Consequently,
multiband transport need not look like the full warp `W(z)`.

High-frequency figures show the four **latent channels**, with blue for negative,
white for zero, and red for positive values. One scale is shared across both
endpoints and all four terms; extreme tails are clipped only for display.
The scale is recorded in `metadata.json`, and unnormalized tensors are saved.
High-frequency plots are not decoded photographs. VAE decoding is nonlinear,
so decoded low/high images cannot be added to reconstruct a decoded result.

The default visualizer uses the VAE posterior mean to isolate transport from
sampling noise. Use `run.py --save-latents` followed by `--latents-dir` to load
exact sampled endpoints from a generation run. Input and correspondence hashes
are checked before those tensors are used. The metadata records whether the
reconstructed draft exactly matches the saved draft.

## Draft and midpoint

The first three frames use source coordinates and the last three use target
coordinates. Their displacement strength is 1.0. The middle draft uses 0.75
strength in both endpoint maps, builds the two chart blends, then transports the
target-chart blend using half of that reduced displacement before Slerp.
The effective extra half displacement is therefore 0.375 of the full map.

At the midpoint, the two operands in `path.png` are **blends**, not separately
warped source and target images. `midpoint.png` expands this computation.

`--strength 0.75` changes the endpoint transport display only. It does not change
the released draft path or generation settings. Use a separate output directory
when comparing display strengths.

## Python interface

```python
from alignmorph.transport import transport_components, build_draft_path

terms = transport_components(latent, pullback_map, reliability)
# terms: original, low, high, warped_low, warped_high,
#        moved_high, retained_high, transported, full_warp, reliability, mapping
path = build_draft_path(source_latent, target_latent, correspondence)
```

`source_to_target` is a pullback map that samples target content in source
coordinates; use `target_to_source` for source content in target coordinates.
These maps act on latent grids. Attention K/V are obtained from original and aligned
endpoint reference states. The visualizer does not resample projected Q/K/V.

Custom pairs use the same `--pairs` and `--correspondence` arguments as `run.py`.

The attention gates use the full reference geometry, independent of the
`--strength` display override. The raw
`aligned_reference_weight` is 0.25 times the per-frame gate, with exact zeros at
the endpoints.
