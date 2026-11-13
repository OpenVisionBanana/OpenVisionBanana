"""Prepare referred-image-segmentation eval data into PNG masks.

Two subcommands:

  reasonseg   Rasterize the ReasonSeg (LISA zip release) LabelMe polygon JSONs.
              Each <stem>.json sits next to <stem>.jpg with polygon ``shapes``
              labelled ``target`` / ``ignore`` / ``flag``. Reproduces LISA's
              ``get_mask_from_json``; writes {0, 1, 255} masks
              (0=background, 1=target, 255=ignore, flag skipped).

  refcocog    Rasterize the RefCOCOg UMD *validation* split from the lichengunc
              ``refer`` release (refs(umd).p + instances.json + train2014 images).
              One binary mask per referring expression (ann_id), value 1=target,
              plus a JSON index mapping ref_id -> sentences / image / mask.

Each dataset is rasterized with its canonical tool: ReasonSeg follows LISA's
cv2.polylines+cv2.fillPoly; RefCOCOg uses pycocotools (frPyObjects/decode), which
defines the COCO ground truth.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from tqdm import tqdm

# Repo root, derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[2]
REFSEG_ROOT = REPO_ROOT / "datasets_archives/refseg"


def _repo_relative(path: Path) -> str:
    """Path as a string relative to REPO_ROOT when possible, else absolute."""
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _lisa_fill(mask: np.ndarray, points, value: int) -> None:
    """Draw a polygon into ``mask`` the way LISA does: 1px outline + fill."""
    # Only ReasonSeg needs OpenCV. RefCOCOg uses pycocotools exclusively and
    # must not require the GUI libraries of an unrelated OpenCV installation.
    import cv2
    pts = np.array([points], dtype=np.int32)
    cv2.polylines(mask, pts, True, value, 1)
    cv2.fillPoly(mask, pts, value)


# --------------------------------------------------------------------------- #
# ReasonSeg
# --------------------------------------------------------------------------- #
def mask_from_json(json_path: Path, height: int, width: int) -> np.ndarray:
    """Build a {0, 1, 255} label mask from a ReasonSeg JSON."""
    try:
        anno = json.loads(json_path.read_text())
    except UnicodeDecodeError:
        anno = json.loads(json_path.read_text(encoding="cp1252"))

    valid = []  # (area, points, label_value)
    for shape in anno["shapes"]:
        label = shape["label"].lower()
        if label == "flag":  # deprecated, meaningless annotation (LISA: exact match)
            continue
        value = 255 if "ignore" in label else 1
        tmp = np.zeros((height, width), dtype=np.uint8)
        _lisa_fill(tmp, shape["points"], 1)
        valid.append((int(tmp.sum()), shape["points"], value))

    # draw largest first so smaller polygons drawn later overwrite the overlap
    valid.sort(key=lambda v: v[0], reverse=True)

    mask = np.zeros((height, width), dtype=np.uint8)
    for _, points, value in valid:
        _lisa_fill(mask, points, value)
    return mask


def prep_reasonseg(args: argparse.Namespace) -> None:
    out_dir = args.out_dir or (args.data_dir / "masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(args.data_dir.glob("*.json"))
    if not json_paths:
        raise SystemExit(f"No .json files found in {args.data_dir}")

    for json_path in tqdm(json_paths, desc="reasonseg"):
        img_path = json_path.with_suffix(".jpg")
        if not img_path.exists():
            tqdm.write(f"skip (no image): {img_path.name}")
            continue
        with Image.open(img_path) as im:
            width, height = im.size
        mask = mask_from_json(json_path, height, width)
        Image.fromarray(mask, mode="L").save(out_dir / f"{json_path.stem}.png")

    print(f"Wrote {len(json_paths)} masks to {out_dir}")


# --------------------------------------------------------------------------- #
# RefCOCOg (UMD split)
# --------------------------------------------------------------------------- #
def _coco_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Decode a COCO segmentation into a 0/1 mask via pycocotools (the canonical GT).

    Handles polygons (list), uncompressed RLE (dict with list ``counts``) and
    compressed RLE.
    """
    if isinstance(segmentation, list):  # polygon(s)
        rle = coco_mask.merge(coco_mask.frPyObjects(segmentation, height, width))
    elif isinstance(segmentation["counts"], list):  # uncompressed RLE
        rle = coco_mask.frPyObjects(segmentation, height, width)
    else:  # already compressed RLE
        rle = segmentation
    return coco_mask.decode(rle).astype(np.uint8)


def prep_refcocog(args: argparse.Namespace) -> None:
    data_dir = args.data_dir
    refs_path = data_dir / "refs(umd).p"
    inst_path = data_dir / "instances.json"
    for p in (refs_path, inst_path, args.images_dir):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    with open(refs_path, "rb") as f:
        try:
            refs = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            refs = pickle.load(f, encoding="latin1")  # pickled under py2
    inst = json.loads(inst_path.read_text())
    anns = {a["id"]: a for a in inst["annotations"]}
    imgs = {i["id"]: i for i in inst["images"]}

    val_refs = [r for r in refs if r["split"] == args.split]
    if not val_refs:
        raise SystemExit(f"no refs with split={args.split!r}")

    out_dir = args.out_dir or (data_dir / f"{args.split}_umd")
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    index = []
    seg_by_img = defaultdict(int)
    for ref in tqdm(val_refs, desc=f"refcocog/{args.split}"):
        ann = anns[ref["ann_id"]]
        img = imgs[ann["image_id"]]
        h, w = img["height"], img["width"]
        mask = _coco_mask(ann["segmentation"], h, w)
        rel = f"masks/{ref['ref_id']}.png"
        Image.fromarray(mask, mode="L").save(out_dir / rel)
        seg_by_img[img["file_name"]] += 1
        index.append(
            {
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
                "image_id": ann["image_id"],
                "file_name": img["file_name"],
                "image_path": _repo_relative(args.images_dir / img["file_name"]),
                "height": h,
                "width": w,
                "category_id": ann["category_id"],
                "sentences": [s["raw"] for s in ref["sentences"]],
                "mask": rel,
            }
        )

    index_path = out_dir / f"refcocog_{args.split}_umd.json"
    index_path.write_text(json.dumps(index, indent=2))
    print(
        f"Wrote {len(index)} masks for {len(seg_by_img)} images to {mask_dir}\n"
        f"Index: {index_path}"
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rs = sub.add_parser("reasonseg", help="ReasonSeg LabelMe JSON -> {0,1,255} masks")
    p_rs.add_argument("--data-dir", type=Path, default=REFSEG_ROOT / "reasonseg_val",
                      help="Dir with ReasonSeg <stem>.jpg / <stem>.json pairs.")
    p_rs.add_argument("--out-dir", type=Path, default=None,
                      help="Mask output dir (default: <data-dir>/masks).")
    p_rs.set_defaults(func=prep_reasonseg)

    p_rg = sub.add_parser("refcocog", help="RefCOCOg UMD val -> binary masks + index")
    p_rg.add_argument("--data-dir", type=Path, default=REFSEG_ROOT / "refcocog",
                      help="Dir with refs(umd).p and instances.json.")
    p_rg.add_argument("--images-dir", type=Path, default=REFSEG_ROOT / "refcocog/train2014",
                      help="Dir with the COCO_train2014_*.jpg images.")
    p_rg.add_argument("--split", default="val", choices=["val", "test", "train"],
                      help="UMD split to prepare (default: val).")
    p_rg.add_argument("--out-dir", type=Path, default=None,
                      help="Output dir (default: <data-dir>/<split>_umd).")
    p_rg.set_defaults(func=prep_refcocog)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
