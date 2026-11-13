from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DepthSample:
    dataset: str
    split: str
    image_path: Path
    depth_path: Path
    focal_length: float
    rel_image_path: Path
    output_stem: str
    sample_id: str
    case: str = "standard"
    mask_path: Optional[Path] = None
    metadata: dict[str, str] = field(default_factory=dict)
