#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization "
    "with no text, labels, legends, or extra objects."
)
NORMAL_PROMPT = (
    "[TASK: SURFACE_NORMAL_ESTIMATION] Generate a dense camera-space surface normal map "
    "of the input scene. Output only a clean RGB normal map with no text, labels, legends, "
    "or extra objects. Encode normal vector components x, y, z in red, green, blue by "
    "mapping [-1, 1] to [0, 255]."
)

DIODE_FX = 886.81


@dataclass(frozen=True)
class DiodeRow:
    domain: str
    index_1based: int
    sample_id: str
    image_path: Path
    depth_path: Path
    mask_path: Path
    normal_path: Path | None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_prepared_tar(out_root: Path, tar_path: Path) -> None:
    tar_path = tar_path.expanduser().resolve()
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tar_path.with_suffix(tar_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    with tarfile.open(tmp_path, "w") as archive:
        archive.add(out_root, arcname=out_root.name)
    tmp_path.replace(tar_path)


def reset_out_root(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_csv_path(value: str) -> Path:
    return Path(value.strip().lstrip("./").lstrip("/"))


def resolve_source_file(source_root: Path, rel_path: Path) -> Path:
    candidates = [
        source_root / rel_path,
        source_root / "full_val" / rel_path,
    ]
    rel_parts = rel_path.parts
    if rel_parts and rel_parts[0] == "val_normal":
        val_rel = Path("val", *rel_parts[1:])
        candidates.extend([source_root / val_rel, source_root / "full_val" / val_rel])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing DIODE source file for {rel_path}. Tried:\n  {tried}")


def sample_id_from_image_path(path: Path) -> str:
    return path.stem


def read_domain_rows(
    source_root: Path,
    domain: str,
    csv_name: str,
    *,
    require_normals: bool,
) -> list[DiodeRow]:
    csv_path = source_root / "data_list" / csv_name
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing DIODE data-list CSV: {csv_path}")

    rows: list[DiodeRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for index, parts in enumerate(csv.reader(handle), start=1):
            if len(parts) != 4:
                raise ValueError(f"Malformed DIODE row in {csv_path}:{index}: {parts}")
            image_rel, depth_rel, mask_rel, normal_rel = (clean_csv_path(part) for part in parts)
            rows.append(
                DiodeRow(
                    domain=domain,
                    index_1based=index,
                    sample_id=sample_id_from_image_path(image_rel),
                    image_path=resolve_source_file(source_root, image_rel),
                    depth_path=resolve_source_file(source_root, depth_rel),
                    mask_path=resolve_source_file(source_root, mask_rel),
                    normal_path=(
                        resolve_source_file(source_root, normal_rel)
                        if require_normals
                        else None
                    ),
                )
            )
    return rows


def load_rows(
    source_root: Path,
    domains: tuple[str, ...],
    *,
    require_normals: bool = True,
) -> list[DiodeRow]:
    domain_to_csv = {
        "indoor": "val_indoors.csv",
        "outdoor": "val_outdoor.csv",
    }
    rows: list[DiodeRow] = []
    for domain in domains:
        rows.extend(
            read_domain_rows(
                source_root,
                domain,
                domain_to_csv[domain],
                require_normals=require_normals,
            )
        )
    return rows


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool).astype(np.uint8) * 255


def load_depth_mask(path: Path, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    raw_mask = np.load(path)
    mask = np.asarray(raw_mask).squeeze().astype(bool)
    if mask.shape != depth.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match depth shape {depth.shape}: {path}")
    return mask & np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


def normals_to_rgb(normals: np.ndarray, valid: np.ndarray, invalid_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.empty((*normals.shape[:2], 3), dtype=np.uint8)
    rgb[...] = np.asarray(invalid_rgb, dtype=np.uint8)
    encoded = np.rint((np.clip(normals, -1.0, 1.0) * 0.5 + 0.5) * 255.0)
    rgb[valid] = np.clip(encoded[valid], 0, 255).astype(np.uint8)
    return rgb


def load_normal(path: Path, valid_depth_mask: np.ndarray, min_normal_norm: float) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(np.load(path), dtype=np.float32).squeeze()
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 normal array at {path}, got {normals.shape}")
    if normals.shape[:2] != valid_depth_mask.shape:
        raise ValueError(f"Normal shape {normals.shape[:2]} does not match mask shape {valid_depth_mask.shape}: {path}")
    norm = np.linalg.norm(normals, axis=-1)
    valid = valid_depth_mask & np.isfinite(norm) & (norm >= min_normal_norm)
    normals = np.divide(normals, norm[..., None], out=np.zeros_like(normals), where=norm[..., None] >= min_normal_norm)
    return normals, valid


def prepare_depth(args: argparse.Namespace, rows: list[DiodeRow]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/indoor", "images/outdoor", "depths/indoor", "depths/outdoor", "masks/indoor", "masks/outdoor", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines = {"indoor": [], "outdoor": []}
    metadata_rows: list[dict] = []
    counts = {"indoor": 0, "outdoor": 0}
    for row in tqdm(rows, desc="Preparing DIODE depth"):
        depth = np.asarray(np.load(row.depth_path), dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth array at {row.depth_path}, got {depth.shape}")
        valid = load_depth_mask(row.mask_path, depth, min_depth=args.min_depth, max_depth=args.max_depth)

        image_rel = Path("images") / row.domain / f"{row.sample_id}.png"
        depth_rel = Path("depths") / row.domain / f"{row.sample_id}_depth.npy"
        mask_rel = Path("masks") / row.domain / f"{row.sample_id}_mask.png"
        shutil.copy2(row.image_path, out_root / image_rel)
        np.save(out_root / depth_rel, depth.astype(np.float32, copy=False))
        Image.fromarray(mask_to_uint8(valid), mode="L").save(out_root / mask_rel)

        split_lines[row.domain].append(
            f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
            f"{args.fx:.10f} {row.sample_id} {row.domain}\n"
        )
        metadata_rows.append(
            {
                "id": row.sample_id,
                "split": "test",
                "case": row.domain,
                "conditioning_image": image_rel.as_posix(),
                "depth_raw": depth_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "task": "depth_estimation",
                "source_dataset": f"diode_{row.domain}",
                "depth_unit": "meters",
                "text": args.depth_prompt,
            }
        )
        counts[row.domain] += 1

    for domain, lines in split_lines.items():
        (out_root / f"diode_{domain}.txt").write_text("".join(lines), encoding="utf-8")
    combined = "".join(split_lines["indoor"] + split_lines["outdoor"])
    (out_root / "diode_all.txt").write_text(combined, encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "diode_depth", counts)
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, rows: list[DiodeRow]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images/indoor", "images/outdoor", "normals/indoor", "normals/outdoor", "masks/indoor", "masks/outdoor", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines = {"indoor": [], "outdoor": []}
    metadata_rows: list[dict] = []
    counts = {"indoor": 0, "outdoor": 0}
    source_index = 0
    for row in tqdm(rows, desc="Preparing DIODE surface normals"):
        if row.normal_path is None:
            raise RuntimeError("Normal preparation requires DIODE normal source files")
        depth = np.asarray(np.load(row.depth_path), dtype=np.float32).squeeze()
        valid_depth = load_depth_mask(row.mask_path, depth, min_depth=args.min_depth, max_depth=args.max_depth)
        normals, normal_valid = load_normal(row.normal_path, valid_depth, min_normal_norm=args.min_normal_norm)
        normal_rgb = normals_to_rgb(normals, normal_valid, args.invalid_normal_rgb)

        image_rel = Path("images") / row.domain / f"{row.sample_id}.png"
        normal_rel = Path("normals") / row.domain / f"{row.sample_id}_normal.png"
        mask_rel = Path("masks") / row.domain / f"{row.sample_id}_normal_mask.png"
        shutil.copy2(row.image_path, out_root / image_rel)
        Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
        Image.fromarray(mask_to_uint8(normal_valid), mode="L").save(out_root / mask_rel)

        source_index += 1
        split_lines[row.domain].append(
            f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} "
            f"{args.fx:.10f} {source_index} {row.sample_id} {row.domain}\n"
        )
        metadata_rows.append(
            {
                "id": row.sample_id,
                "split": "test",
                "case": row.domain,
                "conditioning_image": image_rel.as_posix(),
                "image": normal_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "task": "surface_normal_estimation",
                "source_dataset": f"diode_{row.domain}",
                "normal_ground_truth": "official_diode_normal_npy",
                "normal_encoding": "rgb = round((normal_xyz * 0.5 + 0.5) * 255)",
                "text": args.normal_prompt,
            }
        )
        counts[row.domain] += 1

    for domain, lines in split_lines.items():
        (out_root / f"diode_{domain}.txt").write_text("".join(lines), encoding="utf-8")
    combined = "".join(split_lines["indoor"] + split_lines["outdoor"])
    (out_root / "diode_all.txt").write_text(combined, encoding="utf-8")
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "diode_surface_normals", counts)
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def write_dataset_info(out_root: Path, args: argparse.Namespace, dataset_name: str, counts: dict[str, int]) -> None:
    source_root = args.source_root.expanduser().resolve()
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "geometry_version": (
                "geometry_canonical_v2"
                if dataset_name == "diode_depth" and args.max_depth >= 350.0
                else None
            ),
            "depth_quantity": (
                "projective_camera_z_meters"
                if dataset_name == "diode_depth"
                else None
            ),
            "geometry_audit": (
                str(args.geometry_audit.expanduser().resolve())
                if args.geometry_audit is not None
                else None
            ),
            "source_root": str(source_root),
            "source_archive": (
                str(args.source_archive.expanduser().resolve())
                if args.source_archive is not None
                else None
            ),
            "source_split_files": {
                "indoor": str(source_root / "data_list" / "val_indoors.csv"),
                "outdoor": str(source_root / "data_list" / "val_outdoor.csv"),
            },
            "counts": counts | {"all": counts["indoor"] + counts["outdoor"]},
            "split": "DIODE validation only",
            "cases": ["indoor", "outdoor", "all"],
            "intrinsics": {
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
                "source": "DIODE devkit intrinsics.txt",
            },
            "validity": {
                "min_depth_m": args.min_depth,
                "max_depth_m": args.max_depth,
                "mask": "official DIODE depth mask intersected with finite depth and configured depth range",
            },
            "normal_encoding": "rgb = round((normal_xyz * 0.5 + 0.5) * 255), invalid pixels set to invalid_normal_rgb",
        },
    )


def parse_domains(value: str) -> tuple[str, ...]:
    value = value.lower()
    if value == "all":
        return ("indoor", "outdoor")
    parts = tuple(part.strip() for part in value.replace(",", " ").split() if part.strip())
    unknown = set(parts) - {"indoor", "outdoor"}
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported DIODE domain(s): {', '.join(sorted(unknown))}")
    return parts


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare DIODE validation data for depth and surface-normal evaluation.")
    parser.add_argument("--source-root", type=Path, required=True, help="Directory holding the DIODE validation archives or extracted tree.")
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--domains", type=parse_domains, default=("indoor", "outdoor"))
    parser.add_argument("--depth-out-root", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--normals-out-root", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--depth-tar-output", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--normals-tar-output", type=Path, help="Default: next to --source-root.")
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=300.0)
    parser.add_argument("--min-normal-norm", type=float, default=0.01)
    parser.add_argument("--invalid-normal-rgb", type=lambda value: tuple(int(part) for part in value.split(",")), default=(128, 128, 128))
    parser.add_argument("--fx", type=float, default=DIODE_FX)
    parser.add_argument("--fy", type=float, default=927.06)
    parser.add_argument("--cx", type=float, default=512.0)
    parser.add_argument("--cy", type=float, default=384.0)
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normal-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--geometry-audit", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.depth_out_root = args.depth_out_root or args.source_root / "diode_depth"
    args.normals_out_root = args.normals_out_root or args.source_root / "diode_surface_normals"
    args.depth_tar_output = args.depth_tar_output or args.source_root / "diode_depth.tar"
    args.normals_tar_output = args.normals_tar_output or args.source_root / "diode_surface_normals.tar"
    if len(args.invalid_normal_rgb) != 3 or any(channel < 0 or channel > 255 for channel in args.invalid_normal_rgb):
        raise ValueError("--invalid-normal-rgb must have the form R,G,B with channels in [0, 255]")
    source_root = args.source_root.expanduser().resolve()
    rows = load_rows(
        source_root,
        args.domains,
        require_normals=args.task in {"surface_normals", "both"},
    )
    if args.task in {"depth", "both"}:
        prepare_depth(args, rows)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, rows)
    print("DIODE preparation complete.")
    if args.task in {"depth", "both"}:
        print(f"  depth out root: {args.depth_out_root.expanduser().resolve()}")
        print(f"  depth tar:      {args.depth_tar_output.expanduser().resolve() if args.depth_tar_output else '<none>'}")
    if args.task in {"surface_normals", "both"}:
        print(f"  normals root:   {args.normals_out_root.expanduser().resolve()}")
        print(f"  normals tar:    {args.normals_tar_output.expanduser().resolve() if args.normals_tar_output else '<none>'}")


if __name__ == "__main__":
    main()
