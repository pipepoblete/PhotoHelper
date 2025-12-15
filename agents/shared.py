from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class FaceAnalysisResult:
    image_path: Path
    face_bbox: tuple[int, int, int, int]
    landmarks: dict[str, tuple[float, float]]
    embedding: Sequence[float]
    face_area_ratio: float
    image_size: tuple[int, int]


@dataclass(frozen=True)
class PortraitCropResult:
    crop_box: tuple[int, int, int, int]
    target_size: tuple[int, int]


ProgressCallback = Callable[[str, int, int], None]
