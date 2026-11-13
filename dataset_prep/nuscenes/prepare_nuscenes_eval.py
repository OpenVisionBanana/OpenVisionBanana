#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
from common.surface_normal_encoding import (
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    CANONICAL_NORMAL_CONVENTION,
)

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input outdoor driving scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization with no text, "
    "labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT


@dataclass(frozen=True)
class NuScenesRecord:
    sample_token: str
    camera: str
    image_path: Path
    lidar_path: Path
    camera_intrinsic: np.ndarray
    lidar_to_ego_rotation: np.ndarray
    lidar_to_ego_translation: np.ndarray
    lidar_ego_to_global_rotation: np.ndarray
    lidar_ego_to_global_translation: np.ndarray
    camera_to_ego_rotation: np.ndarray
    camera_to_ego_translation: np.ndarray
    camera_ego_to_global_rotation: np.ndarray
    camera_ego_to_global_translation: np.ndarray


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


def load_json_table(version_root: Path, name: str) -> list[dict]:
    path = version_root / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing nuScenes metadata file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def index_by_token(rows: list[dict]) -> dict[str, dict]:
    return {row["token"]: row for row in rows}


def quaternion_to_rotation(rotation: list[float]) -> np.ndarray:
    w, x, y, z = [float(value) for value in rotation]
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm <= 0:
        raise ValueError(f"Invalid zero-norm quaternion: {rotation}")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def translation(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(3)


def load_scene_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    names = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not names:
        raise ValueError(f"Scene list is empty: {path}")
    return names


def load_records(
    source_root: Path,
    version: str,
    cameras: tuple[str, ...],
    scene_names: set[str] | None = None,
    limit: int | None = None,
) -> tuple[list[NuScenesRecord], int]:
    version_root = source_root / version
    samples = load_json_table(version_root, "sample.json")
    sample_data_rows = load_json_table(version_root, "sample_data.json")
    sample_data = index_by_token(sample_data_rows)
    calibrated_rows = load_json_table(version_root, "calibrated_sensor.json")
    calibrated = index_by_token(calibrated_rows)
    ego_poses = index_by_token(load_json_table(version_root, "ego_pose.json"))
    sensors = index_by_token(load_json_table(version_root, "sensor.json"))

    selected_scene_count = len(load_json_table(version_root, "scene.json"))
    if scene_names is not None:
        scenes = load_json_table(version_root, "scene.json")
        selected_scene_tokens = {row["token"] for row in scenes if row["name"] in scene_names}
        found_scene_names = {row["name"] for row in scenes if row["token"] in selected_scene_tokens}
        missing_scene_names = sorted(scene_names - found_scene_names)
        if missing_scene_names:
            raise ValueError(f"Scene list contains names absent from metadata: {', '.join(missing_scene_names)}")
        samples = [row for row in samples if row["scene_token"] in selected_scene_tokens]
        selected_scene_count = len(selected_scene_tokens)
    if limit is not None:
        samples = samples[:limit]

    channel_by_calibrated_token = {
        row["token"]: sensors[row["sensor_token"]]["channel"]
        for row in calibrated_rows
    }
    sample_data_by_sample: dict[str, dict[str, str]] = {}
    for row in sample_data_rows:
        if not row.get("is_key_frame", False):
            continue
        channel = channel_by_calibrated_token[row["calibrated_sensor_token"]]
        sample_data_by_sample.setdefault(row["sample_token"], {})[channel] = row["token"]

    records: list[NuScenesRecord] = []
    for sample in samples:
        data = sample_data_by_sample.get(sample["token"], {})
        if "LIDAR_TOP" not in data:
            continue
        lidar_sd = sample_data[data["LIDAR_TOP"]]
        lidar_calib = calibrated[lidar_sd["calibrated_sensor_token"]]
        lidar_pose = ego_poses[lidar_sd["ego_pose_token"]]
        lidar_path = source_root / lidar_sd["filename"]
        if not lidar_path.is_file():
            raise FileNotFoundError(f"Missing nuScenes LiDAR file: {lidar_path}")

        for camera in cameras:
            if camera not in data:
                continue
            camera_sd = sample_data[data[camera]]
            camera_calib = calibrated[camera_sd["calibrated_sensor_token"]]
            camera_pose = ego_poses[camera_sd["ego_pose_token"]]
            image_path = source_root / camera_sd["filename"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing nuScenes camera image: {image_path}")
            records.append(
                NuScenesRecord(
                    sample_token=sample["token"],
                    camera=camera,
                    image_path=image_path,
                    lidar_path=lidar_path,
                    camera_intrinsic=np.asarray(camera_calib["camera_intrinsic"], dtype=np.float32),
                    lidar_to_ego_rotation=quaternion_to_rotation(lidar_calib["rotation"]),
                    lidar_to_ego_translation=translation(lidar_calib["translation"]),
                    lidar_ego_to_global_rotation=quaternion_to_rotation(lidar_pose["rotation"]),
                    lidar_ego_to_global_translation=translation(lidar_pose["translation"]),
                    camera_to_ego_rotation=quaternion_to_rotation(camera_calib["rotation"]),
                    camera_to_ego_translation=translation(camera_calib["translation"]),
                    camera_ego_to_global_rotation=quaternion_to_rotation(camera_pose["rotation"]),
                    camera_ego_to_global_translation=translation(camera_pose["translation"]),
                )
            )
    if not records:
        raise FileNotFoundError(f"No nuScenes records found for version={version} cameras={cameras}")
    return records, selected_scene_count


def load_lidar_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 5 != 0:
        raise ValueError(f"nuScenes LiDAR file does not contain float32 x/y/z/intensity/ring tuples: {path}")
    return values.reshape(-1, 5)[:, :3].astype(np.float32, copy=False)


def lidar_to_camera(points_lidar: np.ndarray, record: NuScenesRecord) -> np.ndarray:
    points_ego_lidar = points_lidar @ record.lidar_to_ego_rotation.T + record.lidar_to_ego_translation
    points_global = points_ego_lidar @ record.lidar_ego_to_global_rotation.T + record.lidar_ego_to_global_translation
    points_ego_camera = (points_global - record.camera_ego_to_global_translation) @ record.camera_ego_to_global_rotation
    return (points_ego_camera - record.camera_to_ego_translation) @ record.camera_to_ego_rotation


def normals_lidar_to_camera(normals_lidar: np.ndarray, record: NuScenesRecord) -> np.ndarray:
    normals_global = normals_lidar @ record.lidar_to_ego_rotation.T @ record.lidar_ego_to_global_rotation.T
    normals_ego_camera = normals_global @ record.camera_ego_to_global_rotation
    normals_camera = normals_ego_camera @ record.camera_to_ego_rotation
    norm = np.linalg.norm(normals_camera, axis=1, keepdims=True)
    return np.divide(normals_camera, norm, out=np.zeros_like(normals_camera), where=norm > 1e-8)


def project_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    u = intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2]
    return u, v, z


def zbuffer_indices(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_shape: tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    u, v, depth = project_points(points_camera, intrinsic)
    finite_projection = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth)
    px = np.zeros(u.shape, dtype=np.int32)
    py = np.zeros(v.shape, dtype=np.int32)
    px[finite_projection] = np.rint(u[finite_projection]).astype(np.int32)
    py[finite_projection] = np.rint(v[finite_projection]).astype(np.int32)
    valid = (
        finite_projection
        & (depth > min_depth)
        & (depth < max_depth)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    candidate_indices = np.nonzero(valid)[0]
    order = candidate_indices[np.argsort(depth[candidate_indices])]
    occupied = np.zeros((height, width), dtype=bool)
    keep: list[int] = []
    keep_y: list[int] = []
    keep_x: list[int] = []
    for idx in order:
        y = int(py[idx])
        x = int(px[idx])
        if occupied[y, x]:
            continue
        occupied[y, x] = True
        keep.append(int(idx))
        keep_y.append(y)
        keep_x.append(x)
    return (
        np.asarray(keep, dtype=np.int64),
        np.asarray(keep_y, dtype=np.int32),
        np.asarray(keep_x, dtype=np.int32),
        depth,
    )


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool).astype(np.uint8) * 255


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def estimate_lidar_normals(points_lidar: np.ndarray, k_neighbors: int, min_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if cKDTree is None:
        raise ImportError(
            "Preparing nuScenes surface normals requires scipy. Install scipy in the preparation environment "
            "or run TASK=depth to prepare depth only."
        )
    if k_neighbors < min_neighbors:
        raise ValueError("--normal-k-neighbors must be >= --normal-min-neighbors")

    tree = cKDTree(points_lidar)
    k = min(k_neighbors + 1, len(points_lidar))
    _, nn = tree.query(points_lidar, k=k, workers=-1)
    if nn.ndim == 1:
        nn = nn[:, None]

    normals = np.zeros_like(points_lidar, dtype=np.float32)
    valid = np.zeros((len(points_lidar),), dtype=bool)
    for idx, neighbors in enumerate(nn):
        neighbors = neighbors[neighbors != idx]
        if len(neighbors) < min_neighbors:
            continue
        patch = points_lidar[neighbors]
        patch = patch[np.isfinite(patch).all(axis=1)]
        if len(patch) < min_neighbors:
            continue
        centered = patch - patch.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(patch) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, int(np.argmin(eigvals))]
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-8:
            continue
        normals[idx] = (normal / norm).astype(np.float32)
        valid[idx] = True
    return normals, valid


def sample_id(record: NuScenesRecord) -> str:
    return f"{record.sample_token}_{record.camera.lower()}"


def camera_case(camera: str) -> str:
    return camera.lower()


def copy_image(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with Image.open(dst) as image:
        width, height = image.size
    return height, width


def prepare_depth(args: argparse.Namespace, records: list[NuScenesRecord]) -> None:
    out_root = args.depth_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images", "depth", "masks", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines_by_case = {camera_case(camera): [] for camera in args.cameras}
    metadata_rows: list[dict] = []
    counts = {camera_case(camera): 0 for camera in args.cameras}
    for record in tqdm(records, desc="Preparing nuScenes sparse depth"):
        sid = sample_id(record)
        case = camera_case(record.camera)
        image_rel = Path("images") / case / f"{sid}.jpg"
        depth_rel = Path("depth") / case / f"{sid}_depth.npy"
        mask_rel = Path("masks") / case / f"{sid}_valid.png"
        intrinsics_rel = Path("intrinsics") / case / f"{sid}.txt"
        height, width = copy_image(record.image_path, out_root / image_rel)

        points_lidar = load_lidar_points(record.lidar_path)
        points_camera = lidar_to_camera(points_lidar, record)
        keep, ys, xs, depth_values = zbuffer_indices(
            points_camera=points_camera,
            intrinsic=record.camera_intrinsic,
            image_shape=(height, width),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        depth = np.zeros((height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        depth[ys, xs] = depth_values[keep].astype(np.float32, copy=False)
        valid[ys, xs] = True

        for rel_path in (depth_rel, mask_rel, intrinsics_rel):
            (out_root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_root / depth_rel, depth)
        Image.fromarray(mask_to_uint8(valid), mode="L").save(out_root / mask_rel)
        np.savetxt(out_root / intrinsics_rel, record.camera_intrinsic.reshape(1, 9), fmt="%.10f")
        fx = float(record.camera_intrinsic[0, 0])

        split_lines_by_case[case].append(
            f"/{image_rel.as_posix()} /{depth_rel.as_posix()} /{mask_rel.as_posix()} "
            f"/{intrinsics_rel.as_posix()} {fx:.10f} {sid} {case}\n"
        )
        metadata_rows.append(
            {
                "id": sid,
                "sample_token": record.sample_token,
                "split": args.split,
                "case": case,
                "camera": record.camera,
                "conditioning_image": image_rel.as_posix(),
                "depth_raw": depth_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "intrinsics": intrinsics_rel.as_posix(),
                "valid_depth_pixels": int(valid.sum()),
                "task": "depth_estimation",
                "source_dataset": f"nuscenes_{args.version}",
                "depth_unit": "meters",
                "text": args.depth_prompt,
            }
        )
        counts[case] += 1

    write_split_files(out_root, "nuscenes", args.split, split_lines_by_case)
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "nuscenes_sparse_depth", counts)
    if args.depth_tar_output is not None:
        write_prepared_tar(out_root, args.depth_tar_output)


def prepare_normals(args: argparse.Namespace, records: list[NuScenesRecord]) -> None:
    out_root = args.normals_out_root.expanduser().resolve()
    reset_out_root(out_root, args.overwrite)
    for rel in ("images", "normals", "masks", "intrinsics", "splits"):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    split_lines_by_case = {camera_case(camera): [] for camera in args.cameras}
    metadata_rows: list[dict] = []
    counts = {camera_case(camera): 0 for camera in args.cameras}
    source_index = 0
    for record in tqdm(records, desc="Preparing nuScenes LiDAR-derived normals"):
        sid = sample_id(record)
        case = camera_case(record.camera)
        image_rel = Path("images") / case / f"{sid}.jpg"
        normal_rel = Path("normals") / case / f"{sid}_normal.png"
        mask_rel = Path("masks") / case / f"{sid}_normal_valid.png"
        intrinsics_rel = Path("intrinsics") / case / f"{sid}.txt"
        height, width = copy_image(record.image_path, out_root / image_rel)

        points_lidar = load_lidar_points(record.lidar_path)
        normals_lidar, normal_valid_lidar = estimate_lidar_normals(
            points_lidar,
            k_neighbors=args.normal_k_neighbors,
            min_neighbors=args.normal_min_neighbors,
        )
        points_camera = lidar_to_camera(points_lidar, record)
        normals_camera = normals_lidar_to_camera(normals_lidar, record)
        facing_away = np.sum(normals_camera * points_camera, axis=1) > 0
        normals_camera[facing_away] *= -1.0
        normals_camera = CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(
            normals_camera, normal_valid_lidar
        )

        keep, ys, xs, depth_values = zbuffer_indices(
            points_camera=points_camera,
            intrinsic=record.camera_intrinsic,
            image_shape=(height, width),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        keep_valid = normal_valid_lidar[keep] & np.isfinite(normals_camera[keep]).all(axis=1)
        keep = keep[keep_valid]
        ys = ys[keep_valid]
        xs = xs[keep_valid]

        normal_map = np.zeros((height, width, 3), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        normal_map[ys, xs] = normals_camera[keep].astype(np.float32, copy=False)
        valid[ys, xs] = True
        normal_rgb = normals_to_rgb(
            normal_map, valid, args.invalid_normal_rgb, args.normal_encoding
        )

        for rel_path in (normal_rel, mask_rel, intrinsics_rel):
            (out_root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(normal_rgb, mode="RGB").save(out_root / normal_rel)
        Image.fromarray(mask_to_uint8(valid), mode="L").save(out_root / mask_rel)
        np.savetxt(out_root / intrinsics_rel, record.camera_intrinsic.reshape(1, 9), fmt="%.10f")
        fx = float(record.camera_intrinsic[0, 0])
        source_index += 1

        split_lines_by_case[case].append(
            f"/{image_rel.as_posix()} /{normal_rel.as_posix()} /{mask_rel.as_posix()} "
            f"/{intrinsics_rel.as_posix()} {fx:.10f} {source_index} {sid} {case}\n"
        )
        metadata_rows.append(
            {
                "id": sid,
                "sample_token": record.sample_token,
                "split": args.split,
                "case": case,
                "camera": record.camera,
                "conditioning_image": image_rel.as_posix(),
                "image": normal_rel.as_posix(),
                "mask": mask_rel.as_posix(),
                "intrinsics": intrinsics_rel.as_posix(),
                "valid_normal_pixels": int(valid.sum()),
                "normal_ground_truth": "lidar_top_3d_neighbors_projected_sparse",
                "normal_frame": CANONICAL_NORMAL_CONVENTION,
                "normal_orientation": "face_camera",
                "normal_encoding": args.normal_encoding,
                "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
                "normal_transform": (
                    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                ),
                "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
                "task": "surface_normal_estimation",
                "source_dataset": f"nuscenes_{args.version}",
                "text": args.normal_prompt,
            }
        )
        counts[case] += 1

    write_split_files(out_root, "nuscenes", args.split, split_lines_by_case)
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_dataset_info(out_root, args, "nuscenes_lidar_surface_normals", counts)
    if args.normals_tar_output is not None:
        write_prepared_tar(out_root, args.normals_tar_output)


def write_split_files(out_root: Path, prefix: str, split: str, split_lines_by_case: dict[str, list[str]]) -> None:
    combined: list[str] = []
    for case, lines in sorted(split_lines_by_case.items()):
        (out_root / f"{prefix}_{case}.txt").write_text("".join(lines), encoding="utf-8")
        combined.extend(lines)
    (out_root / f"{prefix}_{split}.txt").write_text("".join(combined), encoding="utf-8")
    (out_root / f"{prefix}_all.txt").write_text("".join(combined), encoding="utf-8")


def write_dataset_info(out_root: Path, args: argparse.Namespace, dataset_name: str, counts: dict[str, int]) -> None:
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source_dataset": "nuScenes",
            "source_root": str(args.source_root.expanduser().resolve()),
            "version": args.version,
            "split": args.split,
            "scene_count": args.scene_count,
            "cases": sorted(counts),
            "counts": counts | {"all": int(sum(counts.values()))},
            "cameras": list(args.cameras),
            "lidar": "LIDAR_TOP keyframe only",
            "depth_unit": "meters",
            "validity": {
                "min_depth_m": args.min_depth,
                "max_depth_m": args.max_depth,
                "mask": "pixels where a projected LIDAR_TOP point survived camera projection and z-buffering",
            },
            "normal_ground_truth": "sparse LiDAR-derived normals; not official dense nuScenes normal annotations",
            "normal_estimation": {
                "method": "PCA over k nearest LiDAR neighbors before projection",
                "k_neighbors": args.normal_k_neighbors,
                "min_neighbors": args.normal_min_neighbors,
                "frame": CANONICAL_NORMAL_CONVENTION,
                "orientation": "face_camera",
                "native_to_canonical": (
                    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                ),
            },
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            "geometry_version": CAMERA_FACING_GEOMETRY_DATA_VERSION,
            "normal_orientation": "face_camera_ray",
            "stored_normal_convention": CANONICAL_NORMAL_CONVENTION,
            "gt_to_canonical_permutation": [0, 1, 2],
            "gt_to_canonical_signs": [1, 1, 1],
            "notes": (
                "Report these results as nuScenes sparse LiDAR-derived outdoor-driving evaluation. "
                "They are not directly comparable to dense monocular depth or normal benchmarks."
            ),
        },
    )


def parse_cameras(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip().upper() for part in value.replace(",", " ").split() if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("At least one camera is required")
    unknown = set(parts) - {
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    }
    if unknown:
        raise argparse.ArgumentTypeError(f"Unsupported nuScenes camera(s): {', '.join(sorted(unknown))}")
    return parts


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3 or any(part < 0 or part > 255 for part in parts):
        raise argparse.ArgumentTypeError("Expected RGB triplet like 128,128,128")
    return parts[0], parts[1], parts[2]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare nuScenes sparse LiDAR depth and LiDAR-derived normal eval tars.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", choices=("test", "train", "val"))
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--task", choices=("depth", "surface_normals", "both"), default="both")
    parser.add_argument("--cameras", type=parse_cameras, default=("CAM_FRONT",))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-tar-output", type=Path)
    parser.add_argument("--normals-tar-output", type=Path)
    parser.add_argument("--min-depth", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--normal-k-neighbors", type=int, default=20)
    parser.add_argument("--normal-min-neighbors", type=int, default=6)
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument(
        "--normal-encoding",
        default=VISION_BANANA_NORMAL_ENCODING,
        choices=("vision_banana_x_left_red", "legacy_direct_xyz"),
    )
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normal-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.split is None:
        args.split = "val" if args.version == "v1.0-trainval" else "test"
    if args.version == "v1.0-trainval" and args.split not in {"train", "val"}:
        raise ValueError("v1.0-trainval preparation supports --split train or val")
    if args.version == "v1.0-mini" and args.split != "test":
        raise ValueError("v1.0-mini preparation supports --split test only")
    if args.version == "v1.0-trainval" and args.scene_list is None:
        args.scene_list = Path(__file__).resolve().parent / "splits" / "v1.0-trainval-val-scenes.txt"
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    args.source_root = args.source_root.expanduser().resolve()
    if not (args.source_root / args.version).is_dir():
        raise FileNotFoundError(f"Missing nuScenes version metadata directory: {args.source_root / args.version}")
    if not (args.source_root / "samples").is_dir():
        raise FileNotFoundError(f"Missing nuScenes samples directory: {args.source_root / 'samples'}")

    scene_names = load_scene_names(args.scene_list.expanduser().resolve() if args.scene_list else None)
    records, args.scene_count = load_records(
        args.source_root,
        version=args.version,
        cameras=args.cameras,
        scene_names=scene_names,
        limit=args.limit,
    )
    if args.version == "v1.0-trainval" and args.split == "val" and args.limit is None:
        if scene_names is None:
            raise ValueError("v1.0-trainval val preparation requires --scene-list")
        if args.scene_count != 150:
            raise ValueError(f"Expected 150 official validation scenes, resolved {args.scene_count}")
        if args.cameras == ("CAM_FRONT",) and len(records) != 6019:
            raise ValueError(f"Expected 6019 CAM_FRONT validation records, resolved {len(records)}")
    if args.task in {"depth", "both"}:
        prepare_depth(args, records)
    if args.task in {"surface_normals", "both"}:
        prepare_normals(args, records)

    print("nuScenes preparation complete.")
    if args.task in {"depth", "both"}:
        print(f"  depth root:   {args.depth_out_root.expanduser().resolve()}")
        print(f"  depth tar:    {args.depth_tar_output.expanduser().resolve() if args.depth_tar_output else '<none>'}")
    if args.task in {"surface_normals", "both"}:
        print(f"  normals root: {args.normals_out_root.expanduser().resolve()}")
        print(f"  normals tar:  {args.normals_tar_output.expanduser().resolve() if args.normals_tar_output else '<none>'}")


if __name__ == "__main__":
    main()
