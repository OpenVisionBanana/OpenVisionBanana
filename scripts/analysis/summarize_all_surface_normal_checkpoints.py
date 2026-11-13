#!/usr/bin/env python3
import argparse
import csv
import glob
import json
from pathlib import Path

DATASETS = ("nyu", "ibims1", "scannet", "vkitti2", "diode", "sintel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--new-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    missing = []
    with args.manifest.open() as f:
        checkpoints = list(csv.DictReader(f, delimiter="\t"))
    for checkpoint in checkpoints:
        source = checkpoint["existing_result_root"]
        if source == "-":
            source = str(args.new_root / checkpoint["label"] / "training_campaign_short_guidance3p5")
        for dataset in DATASETS:
            root = Path(source) / dataset
            candidates = glob.glob(str(root / "*_results.json"))
            if len(candidates) != 1:
                missing.append(f"{checkpoint['label']}/{dataset}: result JSON count={len(candidates)}")
                continue
            try:
                payload = json.load(open(candidates[0]))
                assert payload["normal_evaluation"]["protocol_version"] == "geometry-eval-v6"
                assert payload["images_evaluated"] > 0 and payload["images_skipped"] == 0
                metrics = payload["marigold_v1_1_metrics"]
                archive = root / "pred_rgb_raw.tar"
                assert archive.stat().st_size > 1024
            except Exception as exc:
                missing.append(f"{checkpoint['label']}/{dataset}: invalid: {exc}")
                continue
            rows.append({
                "family": checkpoint["family"], "step": int(checkpoint["step"]),
                "checkpoint": checkpoint["label"], "dataset": dataset,
                "images": payload["images_evaluated"],
                "mean_angular_error": metrics["mean_angular_error"],
                "median_angular_error": metrics["median_angular_error"],
                "sub11_25": metrics["sub11_25_error"], "sub22_5": metrics["sub22_5_error"],
                "sub30": metrics["sub30_error"], "result_json": candidates[0],
                "reused_existing": checkpoint["existing_result_root"] != "-",
            })
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "all_results.json").write_text(json.dumps({"results": rows, "missing": missing}, indent=2) + "\n")
    fields = list(rows[0]) if rows else ["family", "step", "checkpoint", "dataset"]
    with (args.output_root / "all_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    by_checkpoint = {c["label"]: {} for c in checkpoints}
    for row in rows:
        by_checkpoint[row["checkpoint"]][row["dataset"]] = row["mean_angular_error"]
    md = ["# Full surface-normal checkpoint sweep", "", "Mean angular error (degrees; lower is better).", "",
          "| checkpoint | " + " | ".join(DATASETS) + " |", "|---|" + "---:|" * len(DATASETS)]
    for checkpoint in checkpoints:
        values = by_checkpoint[checkpoint["label"]]
        md.append("| " + checkpoint["label"] + " | " + " | ".join(f"{values[d]:.3f}" if d in values else "PENDING" for d in DATASETS) + " |")
    md += ["", f"Complete results: {len(rows)}/{len(checkpoints) * len(DATASETS)}.", ""]
    if missing:
        md += ["## Missing or invalid", ""] + [f"- {item}" for item in missing] + [""]
    (args.output_root / "SUMMARY.md").write_text("\n".join(md))
    print(f"complete={len(rows)}/{len(checkpoints) * len(DATASETS)} missing={len(missing)}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
