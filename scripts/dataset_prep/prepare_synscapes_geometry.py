#!/usr/bin/env python3
"""Run the existing Synscapes depth and normal converters together."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-side", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    from common.node_scratch import workspace
    with workspace("synscapes") as scratch:
        prepare(a, scratch)


def prepare(a, scratch):
    source = ["--synscapes-root", str(a.raw_dir)]
    # Preserve extracted installations, but use the existing TAR readers for
    # packed sources. The converter algorithms and target settings do not change.
    if a.raw_dir.is_file():
        source = ["--synscapes-tar", str(a.raw_dir)]
    elif not (a.raw_dir / "Synscapes/img/rgb").is_dir() and not (a.raw_dir / "img/rgb").is_dir():
        archive = a.raw_dir / "synscapes.tar"
        if archive.is_file():
            source = ["--synscapes-tar", str(archive)]
    for task, script in (("Depth", "depth/prepare_synscapes_depth.py"), ("Normals", "surface_normals/prepare_synscapes_surface_normals.py")):
        output = a.output / task / "synscapes.tar"
        command = [sys.executable, str(ROOT / "dataset_prep" / script), *source,
                   "--out-root", str(scratch / task / "synscapes"), "--prepared-tar", str(output),
                   "--seed", str(a.seed), "--num-workers", str(a.workers), "--max-side", str(a.max_side)]
        if task == "Depth":
            command += ["--depth-encoding", "vision_banana_rgb"]
        else:
            command += ["--normal-encoding", "vision_banana_x_left_red", "--write-raw-normals"]
        if a.limit:
            command += ["--limit", str(a.limit)]
        if a.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
