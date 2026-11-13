# Surface normal data preparation

Normals share raw geometry sources with depth. Start with the
[ZS/NZS recipes](../../DATASET.md#complete-zs-and-nzs-recipes) and
[manual access instructions](../../DATASET.md#manual-geometry-downloads).

Prepared versions contain `train/Normals/*.tar` and `eval/Normals/`.
Metadata records normal encoding, axes, annotation method and validity masks.
NYUv2 training uses the 795 official training IDs with original GeoNet labels
and paired DSINE RGB. Replica, ScanNet and Virtual KITTI 2 training normals
are derived from depth/calibration, not independent normal annotations.
Evaluation preserves the author targets and bypasses training filters.

Current depth-derived converters orient vectors against their camera rays,
convert to `camera_x_right_y_up_z_toward_camera` and record the v4
camera-facing contract with identity stored-to-canonical axes. The evaluator
reads this metadata automatically. See the
[normal evaluation guide](../../evaluation/surface_normals/README.md).

For older local geometry packages, use the
[installer reference](../../scripts/dataset_prep/README.md#geometry-data).
Maintainer utilities and source-specific wrappers are described in the
[preparation reference](../../scripts/dataset_prep/README.md#normal-maintenance-tools).
