"""Offline CGF1 evaluation for SA-Co/Gold instance segmentation.

For each of the 7 SA-Co/Gold subsets it runs the (vendored) ``CGF1Evaluator``
over the 3 oracle
annotator GTs (a/b/c) and the model's COCO predictions, reporting
``cgF1 / IL_MCC / pmF1`` per subset plus the mean across subsets (the paper's
headline number, Table 8).

Prediction layout (written by inference_flux2_klein_instanceseg.py):
    <pred_root>/<subset>/coco_predictions_segm.json

Usage:
    python evaluation/instanceseg/eval_saco_gold.py \
        --pred_root <dir with per-subset prediction folders> \
        [--gt_folder <dir with gold_*_release_test.json>] \
        [--subsets sa1b_nps metaclip_nps ...] [--iou_type segm|bbox]

It also exposes ``run_eval`` so the inference script can score in-process after
dumping predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cgf1_eval import CGF1Evaluator  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_GT_FOLDER = REPO_ROOT / "datasets_archives/instanceseg/SA-Co-Gold/gt-annotations"

# subset name -> GT file stem (the 7 SA-Co/Gold subsets).
SACO_GOLD_SUBSETS = {
    "metaclip_nps": "metaclip",          # MetaCLIP captioner
    "sa1b_nps": "sa1b",                  # SA-1B captioner
    "crowded": "crowded",               # Crowded
    "fg_food": "fg_food",               # FG Food
    "fg_sports_equipment": "fg_sports_equipment",  # FG Sports
    "attributes": "attributes",         # Attributes
    "wiki_common": "wiki_common",       # Wiki common
}
GT_PARTS = ("a", "b", "c")  # 3 oracle annotators per subset
GT_FILE_TMPL = "gold_{stem}_merged_{part}_release_test.json"


def gt_paths_for(subset: str, gt_folder: Path) -> list[str]:
    stem = SACO_GOLD_SUBSETS[subset]
    return [str(gt_folder / GT_FILE_TMPL.format(stem=stem, part=p)) for p in GT_PARTS]


def pred_path_for(subset: str, pred_root: Path) -> Path:
    return pred_root / subset / "coco_predictions_segm.json"


def evaluate_subset(
    subset: str,
    gt_folder: Path,
    pred_root: Path,
    iou_type: str = "segm",
    verbose: bool = True,
    max_eval_images: Optional[int] = None,
    eval_ids: Optional[set] = None,
) -> dict:
    """Run CGF1 for one subset; return {cgF1, IL_MCC, pmF1, num_eval_images}.

    ``eval_ids`` restricts scoring to a given set of image ids -- used when
    inference ran on a sampled subset (--limit) so the metric covers only the
    queries that were actually run (both positives and gated-out negatives), not
    every GT id (which would be counted as missed)."""
    gt_paths = gt_paths_for(subset, gt_folder)
    pred_path = pred_path_for(subset, pred_root)
    if not pred_path.is_file():
        raise FileNotFoundError(f"predictions not found for subset '{subset}': {pred_path}")

    evaluator = CGF1Evaluator(gt_path=gt_paths, verbose=verbose, iou_type=iou_type)
    if eval_ids is not None:
        keep = set(eval_ids)
        evaluator.eval_img_ids = [i for i in evaluator.eval_img_ids if i in keep]
    if max_eval_images is not None:  # debug knob: truncate the eval set for quick smoke checks
        evaluator.eval_img_ids = evaluator.eval_img_ids[:max_eval_images]
    summary = evaluator.evaluate(str(pred_path))

    return {
        "cgF1": round(summary[f"cgF1_eval_{iou_type}_cgF1"] * 100, 2),
        "IL_MCC": round(summary[f"cgF1_eval_{iou_type}_IL_MCC"], 2),
        "pmF1": round(summary[f"cgF1_eval_{iou_type}_positive_micro_F1"] * 100, 2),
        "num_eval_images": len(evaluator.eval_img_ids),
    }


def run_eval(
    pred_root: Path,
    gt_folder: Path = DEFAULT_GT_FOLDER,
    subsets: Optional[Iterable[str]] = None,
    iou_type: str = "segm",
    verbose: bool = True,
    max_eval_images: Optional[int] = None,
    out_json: Optional[Path] = None,
    eval_ids_by_subset: Optional[dict] = None,
    provenance: Optional[dict] = None,
) -> dict:
    """Evaluate the requested subsets and return per-subset + aggregate results.

    The aggregate is the mean of each metric across the evaluated subsets (the
    paper's headline). Subsets without a predictions file are skipped with a
    warning so a partial run still reports what it can. ``eval_ids_by_subset``
    optionally restricts each subset's eval set (see evaluate_subset).
    ``provenance`` is merged verbatim into the results JSON (e.g. which presence
    gate produced the predictions -- IL_MCC/cgF1 are gate-driven, so results from
    different gates are not comparable and must stay attributable).
    """
    subsets = list(subsets) if subsets else list(SACO_GOLD_SUBSETS.keys())
    eval_ids_by_subset = eval_ids_by_subset or {}
    per_subset: dict[str, dict] = {}
    for subset in subsets:
        if subset not in SACO_GOLD_SUBSETS:
            raise ValueError(f"unknown subset '{subset}'. Known: {list(SACO_GOLD_SUBSETS)}")
        print(f"\n=== Processing subset: {subset} ===")
        try:
            per_subset[subset] = evaluate_subset(
                subset, gt_folder, pred_root, iou_type, verbose, max_eval_images,
                eval_ids=eval_ids_by_subset.get(subset),
            )
        except FileNotFoundError as exc:
            print(f"  SKIP: {exc}")

    aggregate = None
    if per_subset:
        keys = ("cgF1", "IL_MCC", "pmF1")
        aggregate = {
            k: round(sum(s[k] for s in per_subset.values()) / len(per_subset), 2) for k in keys
        }
        aggregate["num_subsets"] = len(per_subset)

    results = {
        "benchmark": "saco_gold",
        "iou_type": iou_type,
        **(provenance or {}),
        "subsets": per_subset,
        "aggregate": aggregate,
    }

    # Print a compact summary table.
    print("\n" + "=" * 56)
    print(f"SA-Co/Gold instance segmentation  (IoU type: {iou_type})")
    print("=" * 56)
    print(f"{'subset':24} {'cgF1':>7} {'IL_MCC':>7} {'pmF1':>7}")
    for subset, s in per_subset.items():
        print(f"{subset:24} {s['cgF1']:>7} {s['IL_MCC']:>7} {s['pmF1']:>7}")
    if aggregate:
        print("-" * 48)
        print(f"{'aggregate (mean)':24} {aggregate['cgF1']:>7} {aggregate['IL_MCC']:>7} {aggregate['pmF1']:>7}")
    print("=" * 56)

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, indent=2))
        print(f"Saved results to: {out_json}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_root", type=Path, required=True,
                   help="Directory holding per-subset prediction folders "
                        "(<pred_root>/<subset>/coco_predictions_segm.json).")
    p.add_argument("--gt_folder", type=Path, default=DEFAULT_GT_FOLDER,
                   help="Directory with the gold_*_release_test.json GT files.")
    p.add_argument("--subsets", nargs="+", default=None,
                   choices=sorted(SACO_GOLD_SUBSETS.keys()),
                   help="Subset(s) to evaluate (default: all 7).")
    p.add_argument("--iou_type", choices=("segm", "bbox"), default="segm",
                   help="segm for segmentation (default), bbox for detection numbers.")
    p.add_argument("--out_json", type=Path, default=None,
                   help="Where to write saco_gold_results.json (default: <pred_root>/saco_gold_results.json).")
    p.add_argument("--max_eval_images", type=int, default=None,
                   help="DEBUG only: truncate each subset's eval set to the first N image ids.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-image progress bars.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_json = args.out_json or (args.pred_root / "saco_gold_results.json")
    # Standalone re-scores keep the gate attribution: the inference pass records
    # which presence gate produced these predictions in inference_config.json.
    provenance = None
    cfg_path = args.pred_root / "inference_config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        provenance = {k: cfg.get(k) for k in ("gate", "gate_file") if k in cfg}
    run_eval(
        pred_root=args.pred_root,
        gt_folder=args.gt_folder,
        subsets=args.subsets,
        iou_type=args.iou_type,
        verbose=not args.quiet,
        max_eval_images=args.max_eval_images,
        out_json=out_json,
        provenance=provenance,
    )


if __name__ == "__main__":
    main()
