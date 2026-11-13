# Surface normal evaluation

Use `scripts/evaluate_normals.py` with the commands in
[EVALUATION.md](../../EVALUATION.md#marigold-v2-depth-and-corrected-normal-targets).
The public frontend uses guidance 3.5 and records `geometry-eval-v6`.

Predictions use `vision_banana_x_left_red`, an RGB encoding of normal vectors.
A coordinate convention defines vector axes; an encoding maps vectors to image
channels. Ground-truth encoding and axes
come from `dataset_info.json` or `manifest.json`. Float-XYZ targets already
contain physical vectors; do not flip signs based on the dataset name.
Ambiguous pre-v6 RGB archives are rejected.

NYUv2, iBims-1, DIODE, ScanNet, VKITTI2 and Sintel preserve upstream float-XYZ
targets and masks. KITTI and ETH3D use depth-derived targets; nuScenes uses
sparse LiDAR-derived targets. Report derived-target results separately.
KITTI Eigen is depth-only.

Primary metrics use signed camera-space angular errors over the full evaluation
and GT-valid mask: mean, median, RMSE, and fractions below 11.25, 22.5 and
30 degrees. Invalid predictions receive 90 degrees and remain in the denominator.
`marigold_v1_1_metrics` uses Marigold's per-image aggregation and percentage units.
Global-pixel pooled, valid-only, bidirectional, convention-search and eroded-mask
results are diagnostics.

Select one dataset and add `--limit 2 --dry-run` to check data discovery.
Remove `--dry-run` to run predictions. The resolved coordinate contract is
printed during dry runs and saved with the scores.

## Legacy shell wrappers

Older package layouts, guidance-1.0 defaults, resume settings and explicit
legacy-axis overrides are in the
[shell-wrapper reference](../../scripts/evaluation/README.md#legacy-geometry-shell-wrappers).
