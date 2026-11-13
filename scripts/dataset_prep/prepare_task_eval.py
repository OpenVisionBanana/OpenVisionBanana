#!/usr/bin/env python3
"""Materialize one segmentation/editing/generation benchmark for its evaluator.

Writes prepared/<version>/eval/<Task>/<dataset>/ in the layout the matching
scripts/evaluate_*.py front end reads. Trees that are already in that layout
are linked (relative symlinks into raw/); everything else is built once and
marked complete, so reruns are cheap.

  cityscapes      leftImg8bit/, gtFine/ (with *_labelTrainIds.png)
  coco_object_80  images/val2017/, annotations/... (extracted from coco.tar)
  saco_gold       gt-annotations/, metaclip-images/, sa1b-images/
  refcocog        val_umd/refcocog_val_umd.json + masks/, train2014/
  reasonseg       <stem>.jpg/.json pairs + masks/
  imgedit_bench   singleturn/, hard/, multiturn/, ImgEdit_Judge/
  genai_bench     prompts.jsonl
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MARKER = ".ovb-eval-complete"


def link(target: Path, link_path: Path) -> None:
    """Point link_path at target with a relative symlink (replacing a stale one)."""
    if not target.exists():
        raise FileNotFoundError(f"missing raw source: {target}")
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"refusing to replace directory {link_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def count(directory: Path, pattern: str) -> int:
    return sum(1 for _ in directory.rglob(pattern))


def require(name: str, found: int, expected: int) -> None:
    print(f"   {name}: {found}")
    if found != expected:
        raise SystemExit(f"ERROR: expected {expected} for {name}, found {found}")


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO)


def cityscapes(raw: Path, out: Path) -> None:
    for name in ("leftImg8bit", "gtFine"):
        link(raw / name, out / name)
    require("leftImg8bit/val images", count(out / "leftImg8bit/val", "*_leftImg8bit.png"), 500)
    require("gtFine/val labelTrainIds", count(out / "gtFine/val", "*_labelTrainIds.png"), 500)


def coco_object_80(raw: Path, out: Path) -> None:
    archive = raw / "coco.tar"
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    with tarfile.open(archive) as handle:
        members = []
        for member in handle:
            # coco.tar is rooted at dataset/; drop that prefix.
            parts = Path(member.name).parts
            if not parts or parts[0] != "dataset" or len(parts) == 1:
                continue
            if member.issym() or member.islnk():
                raise SystemExit(f"ERROR: links are not expected in {archive}: {member.name}")
            member.name = str(Path(*parts[1:]))
            target = (out / member.name).resolve()
            if root not in target.parents:
                raise SystemExit(f"ERROR: unsafe path in {archive}: {member.name}")
            members.append(member)
        handle.extractall(out, members=members)  # noqa: S202 - paths checked above.
    require("images/val2017", count(out / "images/val2017", "*.jpg"), 5000)
    require("annotations/val2017", count(out / "annotations/val2017", "*.png"), 5000)


def saco_gold(raw: Path, out: Path) -> None:
    for name in ("gt-annotations", "metaclip-images", "sa1b-images"):
        link(raw / name, out / name)


def refcocog(raw: Path, out: Path) -> None:
    link(raw / "train2014", out / "train2014")
    data_dir = next(path.parent for path in sorted(raw.rglob("refs(umd).p")))
    run([sys.executable, "dataset_prep/refseg/prepare_refseg_eval.py", "refcocog",
         "--data-dir", str(data_dir), "--images-dir", str(raw / "train2014"),
         "--split", "val", "--out-dir", str(out / "val_umd")])


def refcocog_packed(raw: Path, target: Path, overwrite: bool = False) -> None:
    """Run the same converter on val-only scratch images, publish actual bytes."""
    from common.refcocog_packed import validation_images, publish_copy
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing packed evaluation data: {target}")
    with validation_images(raw) as (data_dir, prepared):
        run([sys.executable, "dataset_prep/refseg/prepare_refseg_eval.py", "refcocog",
             "--data-dir", str(data_dir), "--images-dir", str(prepared / "train2014"),
             "--split", "val", "--out-dir", str(prepared / "val_umd")])
        (prepared / MARKER).touch()
        archive_path = prepared.parent / "refcocog.tar"
        with tarfile.open(archive_path, "w", dereference=True) as archive:
            archive.add(prepared, arcname="refcocog")
        if overwrite:
            # Publish a complete replacement atomically, never unlink the old TAR first.
            target.parent.mkdir(parents=True, exist_ok=True)
            import tempfile
            handle, name = tempfile.mkstemp(prefix=".refcocog-", suffix=".partial", dir=target.parent)
            os.close(handle)
            partial = Path(name)
            try:
                shutil.copyfile(archive_path, partial)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
        else:
            publish_copy(archive_path, target)


def reasonseg(raw: Path, out: Path) -> None:
    source = raw / "reasonseg_val"
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".json"}:
            target = out / path.name
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)
    run([sys.executable, "dataset_prep/refseg/prepare_refseg_eval.py", "reasonseg", "--data-dir", str(out)])
    require("ReasonSeg val masks", count(out / "masks", "*.png"), 200)


def imgedit_bench(raw: Path, out: Path) -> None:
    benchmark = raw / "Benchmark"
    for path in sorted(benchmark.iterdir()):
        if not path.name.startswith("."):
            link(path, out / path.name)
    link(raw / "ImgEdit_Judge", out / "ImgEdit_Judge")


def genai_bench(raw: Path, out: Path) -> None:
    run([sys.executable, "evaluation/image_generation/genai_bench/prepare_prompts.py",
         "--source_dir", str(raw), "--out_dir", str(out), "--prompts_name", "prompts.jsonl", "--overwrite"])


BUILDERS = {
    "cityscapes": cityscapes,
    "coco_object_80": coco_object_80,
    "saco_gold": saco_gold,
    "refcocog": refcocog,
    "reasonseg": reasonseg,
    "imgedit_bench": imgedit_bench,
    "genai_bench": genai_bench,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--raw-dir", type=Path, required=True, help="The dataset's raw/<Group>/<dataset> directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="prepared/<version>/eval/<Task>/<dataset>")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw, out = args.raw_dir.expanduser().resolve(), args.out_dir.expanduser().absolute()
    if args.dataset == "refcocog" and (raw / "train2014.zip").is_file() and not out.exists():
        refcocog_packed(raw, out.with_suffix(".tar"), args.overwrite)
        print(f"refcocog: packed evaluation data ready at {out.with_suffix('.tar')}")
        return 0
    if args.dataset == "cityscapes" and (raw / "cityscapes.tar").is_file() and not out.exists():
        from common.official_sources import complete_cityscapes_raw_archive
        source = raw / "cityscapes.tar"
        if not complete_cityscapes_raw_archive(source):
            raise ValueError(f"Incomplete original-layout Cityscapes TAR: {source}")
        target = out.with_suffix(".tar")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if os.path.samefile(source, target):
                return 0
            raise FileExistsError(f"Refusing to replace another packed dataset: {target}")
        # Source and prepared roots normally share a filesystem. A hard link
        # avoids both a second byte copy and a persistent extracted tree.
        try:
            os.link(source, target)
        except OSError as exc:
            import errno
            if exc.errno != errno.EXDEV:
                raise
            import tempfile
            handle, name = tempfile.mkstemp(prefix=".cityscapes-", suffix=".partial", dir=target.parent)
            os.close(handle)
            temporary = Path(name)
            try:
                shutil.copyfile(source, temporary)
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"cityscapes: packed evaluation source ready at {target}")
        return 0
    marker = out / MARKER
    if marker.is_file() and not args.overwrite:
        print(f"{args.dataset}: evaluation data already prepared at {out}")
        return 0
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"== Preparing {args.dataset} evaluation data -> {out}")
    BUILDERS[args.dataset](raw, out)
    marker.touch()
    print(f"{args.dataset}: ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
