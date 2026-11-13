"""Explicit camera rotation and height in the Replica raw-world frame.

Camera matrices map Habitat camera coordinates (+X right, +Y up, -Z forward)
to the raw Replica mesh world (+Z up, +Y forward).
"""
import math
import numpy as np


def camera_rotation(yaw, pitch, world_up='z'):
    if world_up not in ('z', 'y'):
        raise ValueError('Specify z (raw Replica) or y (historical transformed world)')
    angle = pitch + (math.pi / 2 if world_up == 'z' else 0)
    c, s = math.cos(angle), math.sin(angle)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = (np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) if world_up == 'z'
           else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    return rot @ rx


def raised_position(position, height, world_up='z'):
    if world_up not in ('z', 'y'):
        raise ValueError('Unknown world-up axis')
    result = np.asarray(position, dtype=np.float32).copy()
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError('Invalid navigable point')
    result[2 if world_up == 'z' else 1] += height
    return result
