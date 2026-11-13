# Depth evaluation

Use the public command in
[EVALUATION.md](../../EVALUATION.md#marigold-v2-depth-and-corrected-normal-targets).
Setup supplies the author benchmarks and pinned split lists; no local geometry
package is required.

The PPD benchmarks are NYUv2-654, KITTI-Eigen-652, ETH3D-454, ScanNet-800
and DIODE-771. PPD supplies masks/clipping, DIODE edge filtering and ETH3D
inference at 1008x672 followed by evaluation at 2048x1360. The 27-pair ETH3D
low-resolution archive is not ETH3D-454. The separate `geometry_v5` lane remains
available.

## Alignment and diagnostics

Match the checkpoint's depth encoding and prompt. The public frontend defaults
to relative inverse grayscale and guidance 3.5. PPD's original alignment uses
seeded log-depth RANSAC; `--ppd-primary-alignment auto` selects by encoding:

| Prediction encoding | Primary alignment |
|---|---|
| `relative_inverse_grayscale` | Fit scale/offset to inverse depth, invert and score |
| Fixed metric RGB/Barron | Decode positive metric depth, then log-depth RANSAC |
| Other grayscale encodings | Require an explicit diagnostic selection |

For comparisons with Marigold V2 or RINO, match the alignment method as well as
the PPD splits and masks. Inverse grayscale and log depth use distinct
representations. Alignment is an evaluation setting and requires no retraining.

Read `primary_metrics` together with `depth_evaluation.primary_lane`.
`ppd_split_disparity_affine_metrics` holds disparity-affine results;
`marigold_v2_ppd_metrics` holds log-depth RANSAC results. Historical direct-gray
RANSAC-log scoring is diagnostic and explicitly selected with
`--ppd-primary-alignment log_depth_ransac`. Direct, other affine and held-out
results remain separately labeled. Official iBIMS also reports depth
disagreement, boundary and planarity measures.

For a startup check, select one dataset and add `--limit 2 --dry-run`.
Remove `--dry-run` for a two-image model check, then remove the limit for the
full benchmark. Use a new output directory when changing the protocol.

## Legacy shell wrappers

The older archive layout, guidance-1.0 defaults, suite profiles and resume
controls are documented in the
[shell-wrapper reference](../../scripts/evaluation/README.md#legacy-geometry-shell-wrappers).
