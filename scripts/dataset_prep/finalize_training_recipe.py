#!/usr/bin/env python3
"""Combine verified official geometry and the established non-geometry pool."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common.release_data import sha256
from common.indexed_tar import load_index

ZS_SOURCES = {"hypersim", "matterport3d", "replica", "synscapes", "taskonomy"}
NZS_SOURCES = ZS_SOURCES | {"nyuv2", "diode", "scannet", "vkitti2"}


def ensure_diode_convention_verified():
    # Consult the current source contract, not a cached row's superseded claim.
    from scripts.dataset_prep.prepare_geometry_training import DIODE_NORMAL_TO_CANONICAL
    if DIODE_NORMAL_TO_CANONICAL.geometry_version != "diode_identity_reproduced_v2":
        raise ValueError("DIODE normal convention remains unverified after the multi-scene audit. "
                         "Resolve the audit before exporting the complete NZS training recipe. "
                         "Older cached geometry metadata does not clear this source-level gate.")
    evidence = json.loads((ROOT / "validation/data_setup/20261029/diode_convention_reconciled.json").read_text())
    audit = evidence["expanded_audit"]
    if (evidence.get("status") != "PASS" or audit.get("status") != "pass"
            or not audit.get("selected_normal_transform_is_identity")
            or audit.get("selected_depth_quantity") != "projective_camera_z"):
        raise ValueError("Missing or invalid reproduced DIODE convention evidence")


def combine(root, pool, seed=42, samples_per_epoch=1_000_000):
    if pool not in {"zs", "nzs"}:
        raise ValueError(f"Unsupported training pool: {pool}")
    if pool == "nzs":
        ensure_diode_convention_verified()
    manifests = root / "manifests"
    other_path = manifests / ("zeroshot_pool.json" if pool == "zs" else "nonzeroshot_pool.json")
    geometry_path = manifests / "geometry_training_specs.json"
    other, geometry = (json.loads(p.read_text()) for p in (other_path, geometry_path))
    expected = ZS_SOURCES if pool == "zs" else NZS_SOURCES
    entries = other["datasets"] + geometry["datasets"]
    if not math.isclose(sum(float(e["weight"]) for e in other["datasets"]), .7, abs_tol=1e-8):
        raise ValueError("The established other-task pool must retain mass 0.7")
    for task, mass in (("depth_estimation", .1), ("surface_normals", .2)):
        selected = [e for e in geometry["datasets"] if e["task"] == task]
        if {e["source_dataset"] for e in selected} != expected or len(selected) != len(expected):
            raise ValueError(f"Incomplete or mixed-version geometry sources for {pool}/{task}")
        for entry in selected:
            archive = (manifests / entry["path"]).resolve()
            index = (manifests / entry["index_path"]).resolve()
            rows, _ = load_index(index, archive)
            if not rows:
                raise ValueError(f"Empty geometry source: {index}")
            if any(not row.get("loss_mask") for row in rows):
                raise ValueError(f"Missing exported geometry loss mask: {index}")
            if task == "surface_normals" and entry["source_dataset"] == "diode":
                if any(str(row.get("geometry_version") or (row.get("metadata") or {}).get("geometry_version", ""))
                       .startswith("diode_identity_candidate_unverified") for row in rows):
                    raise ValueError("DIODE normal convention remains unverified after the multi-scene audit. "
                                     "Resolve the audit before exporting the complete NZS training recipe.")
            entry["weight"] = mass / len(selected)
    if not math.isclose(sum(float(e["weight"]) for e in entries), 1., abs_tol=1e-8):
        raise ValueError("Combined sampling probabilities must sum to one")
    result = dict(samples_per_epoch=samples_per_epoch, seed=seed, shuffle=True,
                  sampling_strategy="without_replacement",
                  datasets=entries, recipe=pool, source_sampling="equal within each geometry task",
                  data_shared_by_model_sizes=["4B", "9B"], loss_mask_modes=["auto", "disabled"],
                  stored_depth_encoding="vision_banana_rgb", training_depth_encoding="relative_inverse_grayscale",
                  training_flags={"depth_target_encoding": "relative_inverse_grayscale",
                                  "depth_target_resize_policy": "metric_field_then_encode",
                                  "normal_target_resize_policy": "vector_field_renormalize_then_encode",
                                  "normal_target_encoding": "vision_banana_x_left_red"},
                  geometry_policy="official-base-clean-v1", parent_manifests={p.name: sha256(p) for p in (other_path, geometry_path)})
    output = manifests / f"{pool}_training.json"
    partial = output.with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    partial.replace(output)
    Path(str(output)+".sha256").write_text(sha256(output)+"  "+output.name+"\n")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pool", choices=("zs", "nzs"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-epoch", type=int, default=1_000_000)
    a = p.parse_args()
    print(combine(a.root, a.pool, a.seed, a.samples_per_epoch))
