#!/usr/bin/env python3
"""Freeze the depth affine choice from a held-out NYU candidate evaluation.

The input is a 32-image, train-split evaluation produced with
``--depth_calibration_candidates_only`` and the immutable
``heldout_nyu_train32_v1.jsonl`` manifest. The resulting record can only be
used with the exact model/prompt/inference configuration that generated it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.geometry_protocols import (  # noqa: E402
    select_depth_affine_protocol,
    write_depth_affine_calibration_record,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: per-sample row is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No per-sample rows in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write an immutable held-out NYU depth affine-calibration record."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results_path = args.results.resolve()
    per_sample_path = args.per_sample.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    if not per_sample_path.is_file():
        raise FileNotFoundError(per_sample_path)
    summary = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{results_path}: summary is not an object")

    depth_evaluation = summary.get("depth_evaluation")
    if not isinstance(depth_evaluation, dict) or not depth_evaluation.get(
        "calibration_candidates_only"
    ):
        raise ValueError(
            "Results were not produced with --depth_calibration_candidates_only: "
            f"{results_path}"
        )
    if summary.get("dataset") != "nyu" or summary.get("requested_split") != "train":
        raise ValueError("Held-out depth calibration must be NYU train-split results")
    if summary.get("resolved_splits") != ["train"]:
        raise ValueError(
            "Held-out depth calibration must resolve exactly the train split, got "
            f"{summary.get('resolved_splits')!r}"
        )

    subset = summary.get("sample_subset")
    if not isinstance(subset, dict) or subset.get("protocol_id") != "heldout_nyu_train32_v1":
        raise ValueError(
            "Held-out depth calibration must use the frozen heldout_nyu_train32_v1 "
            "sample manifest"
        )
    expected_count = int(subset.get("sample_count", 0))
    if expected_count != 32:
        raise ValueError(f"Expected the frozen 32-sample subset, got {expected_count}")

    configuration = depth_evaluation.get("configuration")
    configuration_sha256 = depth_evaluation.get("configuration_sha256")
    if not isinstance(configuration, dict) or not isinstance(configuration_sha256, str):
        raise ValueError(f"Missing depth configuration identity in {results_path}")

    rows = read_jsonl(per_sample_path)
    if len(rows) != expected_count or int(summary.get("images_evaluated", 0)) != expected_count:
        raise ValueError(
            f"Expected {expected_count} held-out samples, found "
            f"{len(rows)} rows / {summary.get('images_evaluated')!r} evaluated"
        )
    sample_ids: list[str] = []
    disparity_abs_rel: list[float] = []
    linear_abs_rel: list[float] = []
    for number, row in enumerate(rows, start=1):
        if row.get("split") != "train" or row.get("dataset") != "nyu":
            raise ValueError(f"{per_sample_path}:{number}: row is not an NYU train sample")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{per_sample_path}:{number}: missing sample_id")
        try:
            disparity = float(row["disparity_affine_abs_rel"])
            linear = float(row["linear_affine_abs_rel"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{per_sample_path}:{number}: missing affine candidate AbsRel metrics"
            ) from error
        if not math.isfinite(disparity) or not math.isfinite(linear):
            raise ValueError(f"{per_sample_path}:{number}: non-finite candidate metric")
        sample_ids.append(sample_id)
        disparity_abs_rel.append(disparity)
        linear_abs_rel.append(linear)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"Duplicate sample IDs in {per_sample_path}")

    selected, candidate_summary = select_depth_affine_protocol(
        disparity_abs_rel, linear_abs_rel
    )
    record = write_depth_affine_calibration_record(
        args.output.resolve(),
        selected_affine_space=selected,
        candidate_summary=candidate_summary,
        configuration=configuration,
        calibration_dataset="nyu",
        calibration_split="train",
        sample_ids=sample_ids,
        source_sha256=sha256_file(results_path),
        calibration_subset=subset,
    )
    if record.get("configuration_sha256") != configuration_sha256:
        raise RuntimeError("Calibration record configuration digest does not match candidates")
    print(
        "Wrote held-out depth calibration record: "
        f"{args.output.resolve()}\n"
        f"  selected affine space: {selected}\n"
        f"  candidate mean AbsRel: disparity={candidate_summary['disparity_abs_rel']:.6f}, "
        f"depth={candidate_summary['depth_abs_rel']:.6f}"
    )


if __name__ == "__main__":
    main()
