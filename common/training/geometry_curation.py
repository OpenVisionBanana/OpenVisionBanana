"""Deterministic training-only edge trimming; missing interior labels stay missing."""
from __future__ import annotations

import numpy as np


CONTRACT = 'nzs-edge-curation-v1'


def source_crop_from_row(row):
    box = row.get('training_crop_box')
    if box is None:
        return None
    if row.get('training_crop_contract') != CONTRACT:
        raise ValueError('Unknown training crop contract')
    if len(box) != 4 or any(type(x) is not int for x in box):
        raise ValueError('Crop coordinates must be four integers')
    return tuple(box)


def cropped_size(size, box):
    if box is None:
        return size
    left, top, right, bottom = box
    width, height = size
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f'Invalid training crop {box} for image size {size}')
    return right - left, bottom - top


def crop_source_image(image, box):
    cropped_size(image.size, box)
    return image if box is None else image.crop(box)


def edge_trim(mask, max_edge_fraction=0.25, min_retained_area=0.6,
              max_strip_valid_fraction=0.5, min_side=32):
    """Trim contiguous mostly-invalid edge rows/columns, never chase interior holes.

    A partial edge stripe must cover at least half its row/column by default.
    If a band continues beyond the safety bound, or too little area remains,
    retain the original rectangle and flag it instead of making a partial trim.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.size:
        raise ValueError('Expected a nonempty 2D validity mask')
    if not (0 <= max_edge_fraction < 0.5 and 0 < min_retained_area <= 1
            and 0 <= max_strip_valid_fraction < 1 and min_side > 0):
        raise ValueError('Invalid edge trimming limits')
    height, width = mask.shape
    original = (0, 0, width, height)
    if not mask.any():
        return original, ['empty_validity']

    def band(values):
        count = 0
        for value in values:
            if value > max_strip_valid_fraction:
                break
            count += 1
        return count

    columns, rows = mask.mean(0), mask.mean(1)
    left, right = band(columns), band(columns[::-1])
    top, bottom = band(rows), band(rows[::-1])
    if (max(left, right) > width * max_edge_fraction
            or max(top, bottom) > height * max_edge_fraction):
        return original, ['excessive_edge_band']
    box = (left, top, width-right, height-bottom)
    retained = (width-left-right) * (height-top-bottom) / mask.size
    if (retained < min_retained_area
            or min(width-left-right, height-top-bottom) < min_side):
        return original, ['excessive_crop']
    return box, []


def quality_signals(rgb, target, valid, normals=False):
    """Cheap review signals, not a semantic label-quality certificate."""
    rgb = np.asarray(rgb, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    gray = rgb.mean(-1)
    lap = (gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1]
           + gray[2:, 1:-1] - 4*gray[1:-1, 1:-1])
    signals = {'rgb_std': float(gray.std()), 'laplacian_variance': float(lap.var()),
               'rgb_extreme_fraction': float(((gray <= 1) | (gray >= 254)).mean())}
    flags, hard = [], []
    if gray.std() < 1:
        flags.append('near_constant_rgb')
    if signals['laplacian_variance'] < 1:
        flags.append('low_texture_or_blur')
    if signals['rgb_extreme_fraction'] > 0.99:
        flags.append('near_uniform_black_or_white_rgb')
    if not valid.any():
        hard.append('empty_validity')
    elif normals:
        vectors = target / 127.5 - 1
        lengths = np.linalg.norm(vectors, axis=-1)
        fraction = float((lengths[valid] < 0.05).mean())
        signals['near_zero_valid_normal_fraction'] = fraction
        if fraction > 0.01:
            hard.append('invalid_vectors_inside_normal_mask')
        signals['nonunit_valid_normal_fraction'] = float((np.abs(lengths[valid]-1) > .025).mean())
        if signals['nonunit_valid_normal_fraction'] > .05:
            flags.append('nonunit_encoded_normal_vectors')
        unit = vectors / np.maximum(lengths[..., None], 1e-6)
        angles = []
        for a,b,m in ((unit[:,:-1],unit[:,1:],valid[:,:-1]&valid[:,1:]),
                      (unit[:-1],unit[1:],valid[:-1]&valid[1:])):
            angles.append(np.degrees(np.arccos(np.clip((a*b).sum(-1)[m],-1,1))))
        angles = np.concatenate(angles)
        if angles.size:
            signals['median_neighbor_normal_angle'] = float(np.median(angles))
            signals['neighbor_angle_over45_fraction'] = float((angles>45).mean())
            if signals['median_neighbor_normal_angle'] > 20:
                flags.append('rough_normal_field_at_review_resolution')
    return signals, flags, hard
