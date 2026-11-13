from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import DepthSample
from . import diode, eth3d, ibims1, kitti, kitti_eigen, nuscenes, nyu, scannet, vkitti2


SUPPORTED_DATASETS = (
    "nyu",
    "ibims1",
    "diode",
    "kitti",
    "kitti_eigen",
    "eth3d",
    "nuscenes",
    "vkitti2",
    "scannet",
)


def canonicalize_split_name(dataset: str, split: str) -> str:
    if dataset == "nyu":
        return nyu.canonicalize_split_name(split)
    if dataset == "ibims1":
        return ibims1.canonicalize_split_name(split)
    if dataset == "diode":
        return diode.canonicalize_split_name(split)
    if dataset == "kitti":
        return kitti.canonicalize_split_name(split)
    if dataset == "kitti_eigen":
        return kitti_eigen.canonicalize_split_name(split)
    if dataset == "eth3d":
        return eth3d.canonicalize_split_name(split)
    if dataset == "nuscenes":
        return nuscenes.canonicalize_split_name(split)
    if dataset == "vkitti2":
        return vkitti2.canonicalize_split_name(split)
    if dataset == "scannet":
        return scannet.canonicalize_split_name(split)
    raise ValueError(f"Unsupported depth dataset: {dataset}")


def resolve_dataset_root(dataset: str, data_root: Path) -> Path:
    if dataset == "nyu":
        return nyu.resolve_dataset_root(data_root)
    if dataset == "ibims1":
        return ibims1.resolve_dataset_root(data_root)
    if dataset == "diode":
        return diode.resolve_dataset_root(data_root)
    if dataset == "kitti":
        return kitti.resolve_dataset_root(data_root)
    if dataset == "kitti_eigen":
        return kitti_eigen.resolve_dataset_root(data_root)
    if dataset == "eth3d":
        return eth3d.resolve_dataset_root(data_root)
    if dataset == "nuscenes":
        return nuscenes.resolve_dataset_root(data_root)
    if dataset == "vkitti2":
        return vkitti2.resolve_dataset_root(data_root)
    if dataset == "scannet":
        return scannet.resolve_dataset_root(data_root)
    raise ValueError(f"Unsupported depth dataset: {dataset}")


def build_samples(
    dataset: str,
    dataset_root: Path,
    split: str,
    limit: Optional[int],
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[DepthSample]:
    if dataset == "nyu":
        return nyu.build_samples(dataset_root=dataset_root, split=split, limit=limit)
    if dataset == "ibims1":
        return ibims1.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "diode":
        return diode.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "kitti":
        return kitti.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "kitti_eigen":
        return kitti_eigen.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "eth3d":
        return eth3d.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "nuscenes":
        return nuscenes.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "vkitti2":
        return vkitti2.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    if dataset == "scannet":
        return scannet.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
            cases=cases,
            aux_augmentations=aux_augmentations,
        )
    raise ValueError(f"Unsupported depth dataset: {dataset}")


def load_gt_depth(sample: DepthSample) -> object:
    if sample.dataset == "nyu":
        return nyu.load_gt_depth(sample.depth_path)
    if sample.dataset == "ibims1":
        return ibims1.load_gt_depth(sample)
    if sample.dataset == "diode":
        return diode.load_gt_depth(sample)
    if sample.dataset == "kitti":
        return kitti.load_gt_depth(sample)
    if sample.dataset == "kitti_eigen":
        return kitti_eigen.load_gt_depth(sample)
    if sample.dataset == "eth3d":
        return eth3d.load_gt_depth(sample)
    if sample.dataset == "nuscenes":
        return nuscenes.load_gt_depth(sample)
    if sample.dataset == "vkitti2":
        return vkitti2.load_gt_depth(sample)
    if sample.dataset == "scannet":
        return scannet.load_gt_depth(sample)
    raise ValueError(f"Unsupported depth dataset: {sample.dataset}")


def valid_depth_mask(
    sample: DepthSample,
    depth_m: object,
    min_depth: float,
    max_depth: float,
) -> object:
    if sample.dataset == "nyu":
        return nyu.valid_depth_mask(depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "ibims1":
        return ibims1.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "diode":
        return diode.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "kitti":
        return kitti.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "kitti_eigen":
        return kitti_eigen.valid_depth_mask(
            sample, depth_m, min_depth=min_depth, max_depth=max_depth
        )
    if sample.dataset == "eth3d":
        return eth3d.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "nuscenes":
        return nuscenes.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "vkitti2":
        return vkitti2.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    if sample.dataset == "scannet":
        return scannet.valid_depth_mask(sample, depth_m, min_depth=min_depth, max_depth=max_depth)
    raise ValueError(f"Unsupported depth dataset: {sample.dataset}")
