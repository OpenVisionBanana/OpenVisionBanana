from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NormalSample:
    dataset: str
    split: str
    image_path: Path
    normal_path: Path
    mask_path: Optional[Path]
    focal_length: float
    source_index: Optional[int]
    rel_image_path: Path
    output_stem: str
    sample_id: str
    case: str = "standard"
    metadata: dict[str, str] = field(default_factory=dict)
