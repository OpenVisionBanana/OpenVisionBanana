"""Fit local planes in inverse depth without mixing disconnected surfaces.

Inverse projective depth is affine on a 3D plane. This estimator fits that
field with spatial/depth-aware weights, then converts the plane analytically
to camera-facing normals. It is training supervision, NOT a benchmark GT edit.
"""
from __future__ import annotations

import numpy as np


def plane_normals(depth, *, fx, fy, cx, cy, radius=4, relative_tolerance=.03,
                  absolute_tolerance=.01, min_support=12, min_fraction=.35):
    z = np.asarray(depth, dtype=np.float64)
    if z.ndim != 2 or radius < 1 or fx <= 0 or fy <= 0:
        raise ValueError("Expected positive intrinsics, radius, and a 2D depth field")
    if relative_tolerance <= 0 or absolute_tolerance < 0 or min_support < 3 or not 0 < min_fraction <= 1:
        raise ValueError("Invalid edge-aware plane support parameters")
    valid = np.isfinite(z) & (z > 0)
    safe = np.where(valid, z, 1.)
    padded = np.pad(safe, radius, constant_values=1.)
    padded_valid = np.pad(valid, radius, constant_values=False)
    h, w = z.shape
    matrix = np.zeros((h, w, 3, 3), dtype=np.float64)
    rhs = np.zeros((h, w, 3), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.int32)
    q = 1. / safe
    tolerance = absolute_tolerance + relative_tolerance * safe
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            usable = padded_valid[radius+dy:radius+dy+h, radius+dx:radius+dx+w] & valid
            distance = (neighbor - safe) / tolerance
            usable &= np.abs(distance) <= 1.
            weight = usable * np.exp(-.5 * distance**2 - (dx*dx+dy*dy)/(2.*radius*radius))
            basis = np.array([dx, dy, 1.], dtype=np.float64)
            matrix += weight[..., None, None] * np.outer(basis, basis)
            rhs += (weight / neighbor)[..., None] * basis
            count += usable
    supported = valid & (count >= max(min_support, int(np.ceil(min_fraction*(2*radius+1)**2))))
    # A one-dimensional support set cannot determine a plane.
    det = np.linalg.det(matrix)
    supported &= np.isfinite(det) & (det > 1e-6)
    matrix[~supported] = np.eye(3)
    rhs[~supported] = 0
    coefficients = np.linalg.solve(matrix, rhs[..., None])[..., 0]
    yy, xx = np.mgrid[:h, :w]
    a, b, c = np.moveaxis(coefficients, -1, 0)
    normal = np.stack((a*fx, b*fy, c-a*(xx-cx)-b*(yy-cy)), axis=-1)
    length = np.linalg.norm(normal, axis=-1)
    supported &= np.isfinite(normal).all(axis=-1) & (length > 1e-8)
    normal = np.divide(normal, length[..., None], out=np.zeros_like(normal), where=length[..., None]>1e-8)
    # Plane n dot ray is positive before reversal. Camera facing orientation,
    # then basis change: native x-right/y-down/z-away -> canonical right/up/toward.
    ray_dot = normal[..., 0]*(xx-cx)/fx + normal[..., 1]*(yy-cy)/fy + normal[..., 2]
    normal[ray_dot > 0] *= -1
    normal *= np.array([1., -1., -1.])
    normal[~supported] = 0
    return normal.astype(np.float32), supported


def target_statistics(normals, valid):
    n = np.asarray(normals, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(n).all(axis=-1)
    angles = []
    for axis in (0, 1):
        first = np.take(n, range(n.shape[axis]-1), axis=axis)
        second = np.take(n, range(1, n.shape[axis]), axis=axis)
        mask = np.take(valid, range(valid.shape[axis]-1), axis=axis) & np.take(valid, range(1, valid.shape[axis]), axis=axis)
        dot = np.clip(np.sum(first*second, axis=-1), -1, 1)
        angles.append(np.degrees(np.arccos(dot))[mask])
    values = np.concatenate(angles)
    return {"valid_fraction": float(valid.mean()), "valid_pixels": int(valid.sum()),
            "neighbor_median_degrees": float(np.median(values)) if len(values) else None,
            "neighbor_p95_degrees": float(np.percentile(values,95)) if len(values) else None}
