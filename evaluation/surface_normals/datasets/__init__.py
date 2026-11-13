from __future__ import annotations

from pathlib import Path
from typing import Optional

from .common import NormalSample
from . import diode, eth3d, ibims1, kitti, nuscenes, nyu, scannet, sintel, vkitti2


SUPPORTED_DATASETS = ("nyu", "ibims1", "diode", "kitti", "eth3d", "nuscenes", "vkitti2", "scannet", "sintel")


def canonicalize_split_name(dataset: str, split: str) -> str:
    if dataset == "nyu":
        return nyu.canonicalize_split_name(split)
    if dataset == "ibims1":
        return ibims1.canonicalize_split_name(split)
    if dataset == "diode":
        return diode.canonicalize_split_name(split)
    if dataset == "kitti":
        return kitti.canonicalize_split_name(split)
    if dataset == "eth3d":
        return eth3d.canonicalize_split_name(split)
    if dataset == "nuscenes":
        return nuscenes.canonicalize_split_name(split)
    if dataset == "vkitti2":
        return vkitti2.canonicalize_split_name(split)
    if dataset == "scannet":
        return scannet.canonicalize_split_name(split)
    if dataset == "sintel":
        return sintel.canonicalize_split_name(split)
    raise ValueError(f"Unsupported surface-normal dataset: {dataset}")


def resolve_dataset_root(dataset: str, data_root: Path) -> Path:
    if dataset == "nyu":
        return nyu.resolve_dataset_root(data_root)
    if dataset == "ibims1":
        return ibims1.resolve_dataset_root(data_root)
    if dataset == "diode":
        return diode.resolve_dataset_root(data_root)
    if dataset == "kitti":
        return kitti.resolve_dataset_root(data_root)
    if dataset == "eth3d":
        return eth3d.resolve_dataset_root(data_root)
    if dataset == "nuscenes":
        return nuscenes.resolve_dataset_root(data_root)
    if dataset == "vkitti2":
        return vkitti2.resolve_dataset_root(data_root)
    if dataset == "scannet":
        return scannet.resolve_dataset_root(data_root)
    if dataset == "sintel":
        return sintel.resolve_dataset_root(data_root)
    raise ValueError(f"Unsupported surface-normal dataset: {dataset}")


def build_samples(
    dataset: str,
    dataset_root: Path,
    split: str,
    limit: Optional[int],
    cases: tuple[str, ...],
    aux_augmentations: tuple[str, ...],
) -> list[NormalSample]:
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
    if dataset == "sintel":
        return sintel.build_samples(
            dataset_root=dataset_root,
            split=split,
            limit=limit,
        )
    raise ValueError(f"Unsupported surface-normal dataset: {dataset}")
