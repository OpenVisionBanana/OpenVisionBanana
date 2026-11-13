#!/usr/bin/env python3
"""Adapt official per-scan Matterport ZIPs to the existing converter."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
MODALITIES = ("undistorted_color_images", "undistorted_depth_images", "undistorted_normal_images", "undistorted_camera_parameters")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-tar-root", type=Path, required=True)
    p.add_argument("--scratch-root", type=Path, required=True)
    a, rest = p.parse_known_args()
    source = a.source_tar_root
    if not list(source.glob("*.tar")):
        staged = a.scratch_root / "official_scan_archives"
        scenes = sorted({path.parent for path in source.rglob("undistorted_color_images.zip")})
        if not scenes:
            raise FileNotFoundError("No official Matterport scan ZIPs under " + str(source))
        for scene in scenes:
            paths = [scene / (m + ".zip") for m in MODALITIES]
            for path in paths:
                if not zipfile.is_zipfile(path):
                    raise ValueError(f"Missing or invalid official Matterport modality: {path}")
            staged.mkdir(parents=True, exist_ok=True)
            output = staged / (scene.name + ".tar")
            if not output.exists():
                partial = output.with_suffix(".tar.partial")
                with tarfile.open(partial, "w") as archive:
                    for path in paths:
                        archive.add(path, arcname=path.name, recursive=False)
                partial.replace(output)
        source = staged
    subprocess.run([sys.executable, str(ROOT / "dataset_prep/matterport3d/prepare_matterport3d_depth_normals.py"),
                    "--source-tar-root", str(source), "--scratch-root", str(a.scratch_root / "conversion"), *rest], check=True)


if __name__ == "__main__":
    main()
