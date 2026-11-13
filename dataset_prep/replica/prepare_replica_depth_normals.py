#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import tarfile
import warnings
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from common.geometry_protocol_prompts import VISION_BANANA_NORMAL_PROMPT
from common.replica_geometry import camera_rotation, raised_position
from common.surface_normal_encoding import (
    SUPPORTED_NORMAL_ENCODINGS,
    VISION_BANANA_NORMAL_ENCODING,
    encode_surface_normals_rgb,
    normal_encoding_formula,
)
from dataset_prep.depth.encoding import (
    DEPTH_ENCODING_GRAYSCALE,
    DEPTH_ENCODING_VISION_BANANA_RGB,
    DEPTH_ENCODINGS,
    encode_depth_to_vision_banana_rgb,
    valid_depth_mask,
)
from dataset_prep.geometry_conventions import (
    CAMERA_FACING_GEOMETRY_DATA_VERSION,
    CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL,
    CANONICAL_DEPTH_QUANTITY,
    CANONICAL_NORMAL_CONVENTION,
    GEOMETRY_DATA_VERSION,
    IMAGE_Y_DOWN_Z_AWAY_TO_CANONICAL,
    orient_normals_toward_camera,
    geometry_data_version_for_normal_encoding,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = lambda iterable, **_: iterable


DEPTH_PROMPT = (
    "[TASK: DEPTH_ESTIMATION] Generate a metric depth visualization of the input indoor scene "
    "using the Vision Banana RGB depth encoding. Output only the RGB depth visualization with no "
    "text, labels, legends, or extra objects."
)
NORMAL_PROMPT = VISION_BANANA_NORMAL_PROMPT


@dataclass(frozen=True)
class ReplicaScene:
    name: str
    root: Path
    stage_config: Optional[Path]
    mesh: Optional[Path]
    navmesh: Optional[Path]


@dataclass
class RenderedSample:
    sample_id: str
    scene: ReplicaScene
    split: str
    index: int
    rgb: Image.Image
    depth: np.ndarray
    valid_depth: np.ndarray
    normals: Optional[np.ndarray]
    valid_normals: Optional[np.ndarray]
    normal_source: Optional[str]
    camera_position: list[float]
    yaw_degrees: float
    pitch_degrees: float


def parse_rgb(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB value must have the form R,G,B")
    rgb = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in rgb):
        raise argparse.ArgumentTypeError("RGB channels must be in [0, 255]")
    return rgb


def sanitize_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def stable_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def discover_scenes(replica_root: Path) -> list[ReplicaScene]:
    replica_root = replica_root.expanduser().resolve()
    if not replica_root.is_dir():
        raise FileNotFoundError(f"Replica root does not exist: {replica_root}")

    scenes: list[ReplicaScene] = []
    scene_directories = {path.parent for path in replica_root.rglob("mesh.ply") if path.parent.name != "habitat"}
    for scene_dir in sorted(scene_directories):
        habitat_dir = scene_dir / "habitat"
        stage_config = habitat_dir / "replica_stage.stage_config.json"
        mesh = scene_dir / "mesh.ply"
        semantic_mesh = habitat_dir / "mesh_semantic.ply"
        navmesh = habitat_dir / "mesh_semantic.navmesh"
        preseg_navmesh = habitat_dir / "mesh_preseg_semantic.navmesh"
        scenes.append(
            ReplicaScene(
                name=scene_dir.name,
                root=scene_dir,
                stage_config=stage_config if stage_config.is_file() else None,
                mesh=semantic_mesh if semantic_mesh.is_file() else mesh if mesh.is_file() else None,
                navmesh=navmesh if navmesh.is_file() else preseg_navmesh if preseg_navmesh.is_file() else None,
            )
        )
    return scenes


def select_scenes(
    scenes: list[ReplicaScene],
    *,
    num_scenes: str,
    scene_list: Optional[Path],
) -> list[ReplicaScene]:
    by_name = {scene.name: scene for scene in scenes}
    if scene_list is not None:
        requested = [
            line.strip()
            for line in scene_list.expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        missing = [name for name in requested if name not in by_name]
        if missing:
            raise ValueError(f"Scene list contains unknown Replica scene(s): {missing}")
        selected = [by_name[name] for name in requested]
    elif num_scenes == "all":
        selected = list(scenes)
    else:
        count = int(num_scenes)
        if count <= 0:
            raise ValueError("--num-scenes must be 'all' or a positive integer")
        selected = scenes[:count]

    if not selected:
        raise RuntimeError("No Replica scenes selected.")
    return selected


def scene_load_candidates(scene: ReplicaScene, mode: str) -> list[Path]:
    candidates: list[Path] = []
    if mode in {"auto", "stage_config"} and scene.stage_config is not None:
        candidates.append(scene.stage_config)
    if mode in {"auto", "mesh"} and scene.mesh is not None:
        candidates.append(scene.mesh)
    return candidates


def make_sensor_spec(habitat_sim: Any, uuid: str, sensor_type: Any, args: argparse.Namespace) -> Any:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    spec.resolution = [args.height, args.width]
    spec.hfov = float(args.hfov)
    spec.position = [0.0, 0.0, 0.0]
    return spec


def resolve_scene_dataset_config(scene: ReplicaScene, replica_root: Path, world_up_axis: str) -> Path:
    """Find the closest dataset configuration within the selected Replica root."""
    root = replica_root.expanduser().resolve()
    directory = scene.root.resolve()
    if not directory.is_relative_to(root):
        raise ValueError(f"Replica scene lies outside the selected root: {directory}")
    dataset_config = root / 'replica.scene_dataset_config.json'
    while True:
        candidate = directory / 'replica.scene_dataset_config.json'
        if candidate.is_file():
            dataset_config = candidate
            break
        if directory == root:
            break
        directory = directory.parent
    if world_up_axis == 'z':
        if not dataset_config.is_file():
            raise FileNotFoundError(f'Raw Z-up Replica rendering requires replica.scene_dataset_config.json within {root}; do not silently use Habitat default stage orientation')
        defaults = json.loads(dataset_config.read_text()).get('stages', {}).get('default_attributes', {})
        if defaults.get('up') != [0, 0, 1] or defaults.get('front') != [0, 1, 0]:
            raise ValueError('Replica scene dataset frame differs from the verified raw Z-up/world-Y-forward convention')
    return dataset_config


def create_simulator(scene: ReplicaScene, args: argparse.Namespace, *, want_normals: bool) -> tuple[Any, bool, Path]:
    dataset_config = resolve_scene_dataset_config(scene, args.replica_root, args.camera_world_up_axis)
    import habitat_sim

    normal_type = getattr(habitat_sim.SensorType, "NORMAL", None)
    if want_normals and normal_type is not None:
        normal_attempts = [True]
        if args.normal_source == "auto":
            normal_attempts.append(False)
    else:
        normal_attempts = [False]
    last_error: Optional[Exception] = None

    for enable_normals in normal_attempts:
        for scene_id in scene_load_candidates(scene, args.scene_id_mode):
            try:
                backend = habitat_sim.SimulatorConfiguration()
                backend.scene_id = str(scene_id)
                if dataset_config.is_file():
                    backend.scene_dataset_config_file = str(dataset_config)
                backend.enable_physics = False
                if hasattr(backend, "gpu_device_id"):
                    backend.gpu_device_id = int(args.gpu_device_id)

                agent_cfg = habitat_sim.AgentConfiguration()
                sensors = [
                    make_sensor_spec(habitat_sim, "color_sensor", habitat_sim.SensorType.COLOR, args),
                    make_sensor_spec(habitat_sim, "depth_sensor", habitat_sim.SensorType.DEPTH, args),
                ]
                if enable_normals:
                    sensors.append(make_sensor_spec(habitat_sim, "normal_sensor", normal_type, args))
                agent_cfg.sensor_specifications = sensors

                sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
                if scene.navmesh is not None:
                    try:
                        sim.pathfinder.load_nav_mesh(str(scene.navmesh))
                    except Exception as exc:
                        print(f"WARNING: could not load navmesh for {scene.name}: {exc}", flush=True)
                return sim, enable_normals, scene_id
            except Exception as exc:  # pragma: no cover - depends on Habitat-Sim runtime
                last_error = exc

    raise RuntimeError(f"Could not load Replica scene {scene.name}: {last_error}")


def close_simulator(sim: Any) -> None:
    if hasattr(sim, "close"):
        sim.close()


def make_rotation(yaw_radians: float, pitch_radians: float, world_up: str = 'z') -> Any:
    import quaternion
    return quaternion.from_rotation_matrix(camera_rotation(yaw_radians, pitch_radians, world_up))


def render_observation(sim: Any, position: np.ndarray, yaw: float, pitch: float, world_up: str = 'z') -> dict[str, Any]:
    import habitat_sim

    agent = sim.initialize_agent(0)
    state = habitat_sim.AgentState()
    state.position = position.astype(np.float32)
    state.rotation = make_rotation(yaw, pitch, world_up)
    agent.set_state(state)
    return sim.get_sensor_observations()


def rgb_observation_to_image(obs: Any) -> Image.Image:
    rgb = np.asarray(obs)
    if rgb.ndim != 3:
        raise ValueError(f"Expected RGB observation HxWxC, got shape={rgb.shape}")
    rgb = rgb[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.size and float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def normal_observation_to_array(obs: Any) -> np.ndarray:
    normal = np.asarray(obs)
    if normal.ndim != 3 or normal.shape[-1] < 3:
        raise ValueError(f"Expected normal observation HxWx3/4, got shape={normal.shape}")
    normal = normal[..., :3].astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(obs).dtype, np.integer):
        normal = normal / 127.5 - 1.0
    elif normal.size and float(np.nanmin(normal)) >= 0.0 and float(np.nanmax(normal)) <= 1.0:
        normal = normal * 2.0 - 1.0
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.divide(normal, norm, out=np.zeros_like(normal), where=norm > 1e-8)


def derive_normals_from_depth(depth: np.ndarray, valid: np.ndarray, hfov_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    hfov = math.radians(hfov_degrees)
    fx = 0.5 * width / math.tan(0.5 * hfov)
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    z = depth.astype(np.float32, copy=False)
    points = np.stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z), axis=-1)
    points[~valid] = np.nan

    dx = np.empty_like(points)
    dy = np.empty_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]

    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 1e-8)
    normal_valid = valid & np.isfinite(normals).all(axis=-1) & (np.linalg.norm(normals, axis=-1) > 0.5)
    normals = orient_normals_toward_camera(normals, points, normal_valid)
    return normals.astype(np.float32, copy=False), normal_valid


def orient_normals_z_positive(normals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        return normals
    oriented = normals.copy()
    z_median = float(np.nanmedian(oriented[..., 2][valid]))
    if z_median < 0:
        oriented *= -1.0
    return oriented


def normals_to_rgb(
    normals: np.ndarray,
    valid: np.ndarray,
    invalid_rgb: tuple[int, int, int],
    encoding: str = VISION_BANANA_NORMAL_ENCODING,
) -> np.ndarray:
    return encode_surface_normals_rgb(
        normals, valid, encoding=encoding, invalid_rgb=invalid_rgb
    )


def depth_to_grayscale(
    depth: np.ndarray,
    valid: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    invalid_value: int,
) -> np.ndarray:
    clipped = np.clip(depth, min_depth, max_depth)
    norm = np.log(clipped / min_depth) / np.log(max_depth / min_depth)
    gray = np.rint((1.0 - norm) * 255.0).clip(0, 255).astype(np.uint8)
    gray[~valid] = np.uint8(invalid_value)
    return gray


def make_depth_target(depth: np.ndarray, valid: np.ndarray, args: argparse.Namespace) -> tuple[Image.Image, str]:
    if args.depth_encoding == DEPTH_ENCODING_VISION_BANANA_RGB:
        rgb = encode_depth_to_vision_banana_rgb(depth, valid, invalid_color=args.depth_invalid_rgb)
        return Image.fromarray(rgb, mode="RGB"), "habitat_depth_sensor_metric_then_vision_banana_rgb_encode"
    if args.depth_encoding == DEPTH_ENCODING_GRAYSCALE:
        gray = depth_to_grayscale(
            depth,
            valid,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            invalid_value=args.depth_invalid_rgb[0],
        )
        return Image.fromarray(gray, mode="L").convert("RGB"), "habitat_depth_sensor_metric_then_grayscale_log_encode"
    raise ValueError(f"Unsupported depth encoding: {args.depth_encoding}")


def scene_split_map(scenes: list[ReplicaScene], seed: int, val_ratio: float) -> dict[str, str]:
    names = [scene.name for scene in scenes]
    rng = random.Random(seed)
    rng.shuffle(names)
    num_val = max(1, int(round(len(names) * val_ratio))) if names and val_ratio > 0 else 0
    if num_val >= len(names):
        num_val = max(0, len(names) - 1)
    val_names = set(names[:num_val])
    return {scene.name: "val" if scene.name in val_names else "train" for scene in scenes}


def view_is_acceptable(rgb: Image.Image, depth: np.ndarray, valid: np.ndarray, args: argparse.Namespace) -> bool:
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid_fraction < args.min_valid_fraction:
        return False
    valid_depth = depth[valid]
    if valid_depth.size == 0 or float(np.nanstd(valid_depth)) < args.min_depth_std:
        return False
    rgb_array = np.asarray(rgb, dtype=np.float32)
    if float(rgb_array.mean()) < args.min_rgb_mean:
        return False
    return True


def render_scene(scene: ReplicaScene, args: argparse.Namespace, split: str) -> list[RenderedSample]:
    want_normals = "normals" in args.tasks and args.normal_source != "derived_from_depth"
    sim, normal_sensor_enabled, loaded_scene_id = create_simulator(scene, args, want_normals=want_normals)
    if args.normal_source == "rendered_from_mesh" and not normal_sensor_enabled:
        close_simulator(sim)
        raise RuntimeError("Habitat-Sim build does not expose SensorType.NORMAL; use --normal-source derived_from_depth")

    try:
        if scene.navmesh is None:
            raise RuntimeError(f"Replica scene {scene.name} has no navmesh; cannot sample camera poses.")

        scene_seed = stable_seed(args.seed, scene.name)
        if hasattr(sim, "seed"):
            sim.seed(scene_seed)
        if hasattr(sim.pathfinder, "seed"):
            sim.pathfinder.seed(scene_seed)
        rng = random.Random(scene_seed)
        samples: list[RenderedSample] = []
        max_attempts = max(args.samples_per_scene, 1) * args.max_retry_factor
        scene_token = sanitize_token(scene.name)

        for _ in tqdm(range(max_attempts), desc=f"Rendering Replica {scene.name}", leave=False):
            if len(samples) >= args.samples_per_scene:
                break
            base_position = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
            if base_position.shape != (3,) or not np.isfinite(base_position).all():
                continue
            position = raised_position(base_position, float(args.camera_height), args.camera_world_up_axis)
            yaw = rng.uniform(0.0, 2.0 * math.pi)
            pitch = math.radians(rng.uniform(args.pitch_min, args.pitch_max))

            try:
                obs = render_observation(sim, position, yaw, pitch, args.camera_world_up_axis)
                rgb = rgb_observation_to_image(obs["color_sensor"])
                depth = np.asarray(obs["depth_sensor"], dtype=np.float32)
                if depth.ndim != 2:
                    raise ValueError(f"Expected 2D depth observation, got shape={depth.shape}")
                valid = valid_depth_mask(depth, min_depth=args.min_depth, max_depth=args.max_depth)
                if not view_is_acceptable(rgb, depth, valid, args):
                    continue

                normals = None
                normal_valid = None
                normal_source = None
                if "normals" in args.tasks:
                    if normal_sensor_enabled and "normal_sensor" in obs and args.normal_source != "derived_from_depth":
                        normals = normal_observation_to_array(obs["normal_sensor"])
                        normal_source = "rendered_from_mesh"
                        normal_norm = np.linalg.norm(normals, axis=-1)
                        normal_valid = valid & np.isfinite(normals).all(axis=-1) & (normal_norm > 0.5) & (normal_norm < 1.5)
                    else:
                        normals, normal_valid = derive_normals_from_depth(depth, valid, args.hfov)
                        normal_source = "derived_from_depth"
                    if args.orient_z_positive and normal_source != "derived_from_depth":
                        normals = orient_normals_z_positive(normals, normal_valid)
                    if normal_source == "derived_from_depth":
                        normals = CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.apply(
                            normals, normal_valid
                        )
                    if float(normal_valid.mean()) < args.min_valid_fraction:
                        continue

                index = len(samples)
                sample_id = f"replica_{scene_token}_{index:06d}"
                samples.append(
                    RenderedSample(
                        sample_id=sample_id,
                        scene=scene,
                        split=split,
                        index=index,
                        rgb=rgb,
                        depth=depth,
                        valid_depth=valid,
                        normals=normals,
                        valid_normals=normal_valid,
                        normal_source=normal_source,
                        camera_position=[float(value) for value in position],
                        yaw_degrees=math.degrees(yaw),
                        pitch_degrees=math.degrees(pitch),
                    )
                )
            except Exception as exc:
                if args.strict:
                    raise
                print(f"WARNING: skipped Replica candidate in {scene.name}: {exc}", flush=True)

        if len(samples) < args.samples_per_scene:
            message = (
                f"Replica scene {scene.name} produced {len(samples)}/{args.samples_per_scene} valid samples "
                f"after {max_attempts} attempts. Loaded scene id: {loaded_scene_id}"
            )
            if args.strict:
                raise RuntimeError(message)
            print(f"WARNING: {message}", flush=True)
        if getattr(args, "ptex_renderer", None) and samples:
            apply_ptex_rgb(samples, args)
        return samples
    finally:
        close_simulator(sim)


def apply_ptex_rgb(samples, args):
    """Replace RGB at identical poses after checking Ptex/depth registration."""
    renderer = str(args.ptex_renderer)
    if (args.width, args.height, args.hfov, args.camera_world_up_axis) != (512, 512, 90., "z"):
        raise ValueError("The verified Replica Ptex renderer requires 512x512, 90-degree HFOV and z-up")
    scene = samples[0].scene
    mesh, textures = scene.root / "mesh.ply", scene.root / "textures"
    if not mesh.is_file() or not textures.is_dir():
        raise FileNotFoundError(f"Official Ptex mesh/textures missing in {scene.root}")
    with tempfile.TemporaryDirectory(prefix="replica-ptex-") as temporary:
        stage = Path(temporary)
        poses = []
        for sample in samples:
            rotation = np.diag([1., -1., -1.]) @ camera_rotation(
                math.radians(sample.yaw_degrees), math.radians(sample.pitch_degrees)).T
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = -rotation @ np.asarray(sample.camera_position)
            poses.append(sample.sample_id + "\t" + "\t".join(map(str, matrix.ravel())))
        posefile = stage / "poses.tsv"
        posefile.write_text("\n".join(poses) + "\n")
        subprocess.run([renderer, str(mesh), str(textures), str(posefile), str(stage)], check=True)
        for sample in samples:
            rendered = np.fromfile(stage / (sample.sample_id + ".f32"), dtype="<f4").reshape(512, 512)
            from scipy.ndimage import binary_erosion
            support = binary_erosion(sample.valid_depth, iterations=3)
            hit = support & np.isfinite(rendered) & (rendered > args.min_depth)
            coverage = hit.sum() / max(1, support.sum())
            error = np.quantile(np.abs(rendered[hit]-sample.depth[hit])/sample.depth[hit], .95) if hit.any() else float("inf")
            if hit.sum() < 10000 or coverage < .99 or error >= .02:
                raise ValueError(f"Replica Ptex/depth registration failed: {sample.sample_id}, coverage={coverage}, p95={error}")
            with Image.open(stage / (sample.sample_id + ".ppm")) as image:
                sample.rgb = image.convert("RGB")


def write_depth_sample(sample: RenderedSample, args: argparse.Namespace) -> dict[str, Any]:
    target, target_preparation = make_depth_target(sample.depth, sample.valid_depth, args)
    cond_rel = Path("conditioning_images") / f"{sample.sample_id}_input.png"
    target_rel = Path("target_images") / f"{sample.sample_id}_depth.png"
    depth_rel = Path("depth_raw") / f"{sample.sample_id}.npy"
    mask_rel = Path("valid_masks") / f"{sample.sample_id}.png"

    (args.depth_out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / depth_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.depth_out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    sample.rgb.save(args.depth_out_root / cond_rel)
    target.save(args.depth_out_root / target_rel)
    depth_to_save = sample.depth.astype(np.float32, copy=True)
    depth_to_save[~sample.valid_depth] = np.nan
    np.save(args.depth_out_root / depth_rel, depth_to_save)
    Image.fromarray(sample.valid_depth.astype(np.uint8) * 255, mode="L").save(args.depth_out_root / mask_rel)

    return {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "depth_raw": depth_rel.as_posix(),
        "valid_mask": mask_rel.as_posix(),
        "text": args.depth_prompt,
        "task": "depth_estimation",
        "source_dataset": "replica",
        "depth_encoding": args.depth_encoding,
        "id": sample.sample_id,
        "preprocess_target": "raw_depth_resize_then_encode",
        "metadata": {
            "scene": sample.scene.name,
            "split": sample.split,
            "sample_index": sample.index,
            "camera_position": sample.camera_position,
            "camera_world_up_axis": args.camera_world_up_axis,
            "camera_setup_revision": "explicit-world-up-v1",
            "yaw_degrees": sample.yaw_degrees,
            "pitch_degrees": sample.pitch_degrees,
            "hfov_degrees": args.hfov,
            "width": args.width,
            "height": args.height,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "raw_depth_units": "meters",
            "depth_convention": args.depth_convention,
            "depth_source": "habitat_sim_depth_sensor",
            "rgb_source": "replica_sdk_ptex" if getattr(args, "ptex_renderer", None) else "legacy_vertex_color",
            "geometry_version": GEOMETRY_DATA_VERSION,
            "canonical_depth_quantity": CANONICAL_DEPTH_QUANTITY,
            "target_preparation": target_preparation,
        },
    }


def write_normal_sample(sample: RenderedSample, args: argparse.Namespace) -> dict[str, Any]:
    if sample.normals is None or sample.valid_normals is None or sample.normal_source is None:
        raise ValueError(f"Sample {sample.sample_id} has no normal target.")
    cond_rel = Path("conditioning_images") / f"{sample.sample_id}_input.png"
    target_rel = Path("target_images") / f"{sample.sample_id}_normal.png"
    mask_rel = Path("masks") / f"{sample.sample_id}_mask.png"
    raw_rel = Path("raw_normals") / f"{sample.sample_id}_normals.npz"

    (args.normals_out_root / cond_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.normals_out_root / target_rel).parent.mkdir(parents=True, exist_ok=True)
    (args.normals_out_root / mask_rel).parent.mkdir(parents=True, exist_ok=True)
    sample.rgb.save(args.normals_out_root / cond_rel)
    Image.fromarray(normals_to_rgb(
        sample.normals,
        sample.valid_normals,
        args.invalid_normal_rgb,
        args.normal_encoding,
    ), mode="RGB").save(
        args.normals_out_root / target_rel
    )
    Image.fromarray(sample.valid_normals.astype(np.uint8) * 255, mode="L").save(args.normals_out_root / mask_rel)

    raw_rel_str = None
    if args.write_raw_normals:
        (args.normals_out_root / raw_rel).parent.mkdir(parents=True, exist_ok=True)
        raw_normals = sample.normals.astype(np.float16 if args.raw_normal_dtype == "float16" else np.float32)
        np.savez_compressed(
            args.normals_out_root / raw_rel,
            normals=raw_normals,
            valid_mask=sample.valid_normals.astype(np.bool_),
        )
        raw_rel_str = raw_rel.as_posix()

    row = {
        "conditioning_image": cond_rel.as_posix(),
        "image": target_rel.as_posix(),
        "loss_mask": mask_rel.as_posix(),
        "mask": mask_rel.as_posix(),
        "text": args.normals_prompt,
        "task": "surface_normal_estimation",
        "source_dataset": "replica",
        "id": sample.sample_id,
        "preprocess_target": "bilinear",
        "metadata": {
            "scene": sample.scene.name,
            "split": sample.split,
            "sample_index": sample.index,
            "camera_position": sample.camera_position,
            "yaw_degrees": sample.yaw_degrees,
            "pitch_degrees": sample.pitch_degrees,
            "hfov_degrees": args.hfov,
            "width": args.width,
            "height": args.height,
            "normal_encoding": args.normal_encoding,
            "normal_encoding_formula": normal_encoding_formula(args.normal_encoding),
            "normal_frame": "camera",
            "normal_source": sample.normal_source,
            "camera_world_up_axis": args.camera_world_up_axis,
            "camera_setup_revision": "explicit-world-up-v1",
            "coordinate_convention": (
                CANONICAL_NORMAL_CONVENTION
                if sample.normal_source == "derived_from_depth"
                else "legacy_unverified_habitat_rendered_normal"
            ),
            **(
                CAMERA_Y_DOWN_Z_AWAY_FACING_CAMERA_TO_CANONICAL.metadata()
                if sample.normal_source == "derived_from_depth"
                else {"quarantine_status": "unverified_native_normal_source"}
            ),
            "geometry_version": (
                CAMERA_FACING_GEOMETRY_DATA_VERSION
                if sample.normal_source == "derived_from_depth"
                else "legacy_unverified"
            ),
            "normal_orientation": (
                "face_camera_ray"
                if sample.normal_source == "derived_from_depth"
                else "unverified_native_orientation"
            ),
            "target_preparation": "rendered_or_depth_derived_camera_space_normals",
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
        },
    }
    if raw_rel_str is not None:
        row["raw_normals"] = raw_rel_str
    return row


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_task_outputs(
    out_root: Path,
    tar_path: Optional[Path],
    rows: list[dict[str, Any]],
    *,
    dataset_name: str,
    source_root: Path,
    args: argparse.Namespace,
) -> None:
    train_rows = [row for row in rows if row["metadata"].get("split") == "train"]
    val_rows = [row for row in rows if row["metadata"].get("split") == "val"]
    metadata_rows = rows if args.include_val_in_metadata else train_rows
    write_jsonl(out_root / "metadata.jsonl", metadata_rows)
    write_jsonl(out_root / "manifests" / "train.jsonl", train_rows)
    write_jsonl(out_root / "manifests" / "val.jsonl", val_rows)
    write_json(
        out_root / "dataset_info.json",
        {
            "dataset": dataset_name,
            "source": str(source_root),
            "rows_written": len(rows),
            "metadata_rows": len(metadata_rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "metadata_contains_val": bool(args.include_val_in_metadata),
            "num_scenes": args.num_scenes,
            "samples_per_scene": args.samples_per_scene,
            "seed": args.seed,
            "val_ratio": args.val_ratio,
            "width": args.width,
            "height": args.height,
            "hfov": args.hfov,
            "min_depth": args.min_depth,
            "max_depth": args.max_depth,
            "normal_source": args.normal_source,
            "depth_convention": args.depth_convention,
            "geometry_version": GEOMETRY_DATA_VERSION,
            "canonical_normal_convention": CANONICAL_NORMAL_CONVENTION,
        },
    )
    if tar_path is not None:
        write_prepared_tar(out_root, tar_path)


def clean_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to rebuild.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Replica depth and surface-normal training pairs.")
    parser.add_argument("--ptex-renderer", type=Path,
                        help="Replica SDK renderer built by scripts/dataset_prep/build_replica_ptex.sh")
    parser.add_argument(
        "--allow-vertex-color-rgb", action="store_true",
        help="Legacy reproduction only: acknowledge that this renderer does not load Replica Ptex textures.",
    )
    parser.add_argument("--replica-root", type=Path, required=True, help="Extracted Replica scene root.")
    parser.add_argument("--depth-out-root", type=Path, required=True)
    parser.add_argument("--normals-out-root", type=Path, required=True)
    parser.add_argument("--depth-prepared-tar", type=Path, default=None)
    parser.add_argument("--normals-prepared-tar", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=["depth", "normals"], default=["depth", "normals"])
    parser.add_argument("--num-scenes", default="all", help="'all' or a positive integer from sorted scenes.")
    parser.add_argument("--scene-list", type=Path, default=None, help="Optional newline-delimited scene names.")
    parser.add_argument("--samples-per-scene", type=int, default=500)
    parser.add_argument("--max-retry-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument('--camera-world-up-axis', choices=['z', 'y'], default='z',
                        help='Raw Replica meshes are Z-up. Y is for explicit historical/transformed-world reproduction only.')
    parser.add_argument("--pitch-min", type=float, default=-15.0)
    parser.add_argument("--pitch-max", type=float, default=15.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--min-depth-std", type=float, default=0.05)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--min-rgb-mean", type=float, default=2.0)
    parser.add_argument("--depth-encoding", choices=DEPTH_ENCODINGS, default=DEPTH_ENCODING_VISION_BANANA_RGB)
    parser.add_argument("--depth-invalid-rgb", type=parse_rgb, default=(0, 0, 0))
    parser.add_argument("--invalid-normal-rgb", type=parse_rgb, default=(128, 128, 128))
    parser.add_argument(
        "--normal-encoding",
        choices=SUPPORTED_NORMAL_ENCODINGS,
        default=VISION_BANANA_NORMAL_ENCODING,
    )
    parser.add_argument(
        "--normal-source",
        choices=["auto", "rendered_from_mesh", "derived_from_depth"],
        default="derived_from_depth",
        help=(
            "Canonical v2 defaults to depth-derived normals. Habitat native normals "
            "remain quarantined unless the legacy escape hatch is explicit."
        ),
    )
    parser.add_argument(
        "--depth-convention",
        default=CANONICAL_DEPTH_QUANTITY,
        help="Metadata label for the rendered depth convention. Verify for your Habitat-Sim build.",
    )
    parser.add_argument("--scene-id-mode", choices=["auto", "stage_config", "mesh"], default="auto")
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--depth-prompt", default=DEPTH_PROMPT)
    parser.add_argument("--normals-prompt", default=NORMAL_PROMPT)
    parser.add_argument("--write-raw-normals", action="store_true")
    parser.add_argument("--raw-normal-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--no-orient-z-positive", dest="orient_z_positive", action="store_false")
    parser.set_defaults(orient_z_positive=True)
    parser.add_argument("--include-val-in-metadata", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--legacy-unverified-rendered-normals",
        action="store_true",
        help="Allow unverified Habitat native normals for historical reproduction only.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.allow_vertex_color_rgb and not getattr(args, "ptex_renderer", None):
        raise RuntimeError(
            "This Habitat renderer produces vertex-colour RGB, not Replica Ptex textures. "
            "Build the official SDK Ptex renderer as described in DATASET.md. "
            "For deliberate legacy reproduction only, pass --allow-vertex-color-rgb."
        )
    if not getattr(args, "ptex_renderer", None):
        warnings.warn("Replica RGB uses vertex colours, not the official Ptex textures.", RuntimeWarning)
    elif not args.ptex_renderer.is_file():
        raise FileNotFoundError(args.ptex_renderer)
    if args.samples_per_scene <= 0:
        raise ValueError("--samples-per-scene must be positive")
    if args.max_retry_factor < 1:
        raise ValueError("--max-retry-factor must be >= 1")
    if args.width < 2 or args.height < 2:
        raise ValueError("--width and --height must be at least 2 for depth-derived tangents")
    if args.min_depth <= 0 or args.max_depth <= args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth > 0")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.pitch_max < args.pitch_min:
        raise ValueError("--pitch-max must be >= --pitch-min")
    if (
        "normals" in args.tasks
        and args.normal_source != "derived_from_depth"
        and not args.legacy_unverified_rendered_normals
    ):
        raise RuntimeError(
            "Replica Habitat native normals are not source-wide convention-verified. "
            "Use --normal-source derived_from_depth for geometry_canonical_v2."
        )

    args.replica_root = args.replica_root.expanduser().resolve()
    args.depth_out_root = args.depth_out_root.expanduser().resolve()
    args.normals_out_root = args.normals_out_root.expanduser().resolve()

    if "depth" in args.tasks:
        clean_output(args.depth_out_root, args.overwrite)
    if "normals" in args.tasks:
        clean_output(args.normals_out_root, args.overwrite)

    scenes = select_scenes(discover_scenes(args.replica_root), num_scenes=args.num_scenes, scene_list=args.scene_list)
    split_by_scene = scene_split_map(scenes, args.seed, args.val_ratio)

    depth_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    for scene in tqdm(scenes, desc="Replica scenes"):
        samples = render_scene(scene, args, split=split_by_scene[scene.name])
        for sample in samples:
            if "depth" in args.tasks:
                depth_rows.append(write_depth_sample(sample, args))
            if "normals" in args.tasks:
                normal_rows.append(write_normal_sample(sample, args))

    if "depth" in args.tasks:
        if not depth_rows and not args.allow_empty:
            raise RuntimeError("No Replica depth rows were written.")
        write_task_outputs(
            args.depth_out_root,
            args.depth_prepared_tar,
            depth_rows,
            dataset_name="replica_depth",
            source_root=args.replica_root,
            args=args,
        )
    if "normals" in args.tasks:
        if not normal_rows and not args.allow_empty:
            raise RuntimeError("No Replica surface-normal rows were written.")
        write_task_outputs(
            args.normals_out_root,
            args.normals_prepared_tar,
            normal_rows,
            dataset_name="replica_surface_normals",
            source_root=args.replica_root,
            args=args,
        )


if __name__ == "__main__":
    main()
