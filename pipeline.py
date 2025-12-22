from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from agents import (
    IdentityGroupingAgent,
    PortraitCroppingAgent,
)
from agents.shared import (
    FaceAnalysisResult,
    PortraitCropResult,
    ProgressCallback,
)


@dataclass(frozen=True)
class SelectedFace:
    analysis: FaceAnalysisResult
    crop: PortraitCropResult


@dataclass(frozen=True)
class IdentityBatch:
    person_id: str
    selected_faces: list[SelectedFace]


@dataclass(frozen=True)
class PipelineSummary:
    batches: list[IdentityBatch]
    total_images: int
    skipped_files: list[Path]


def run_pipeline(
    file_paths: Iterable[Path | str],
    output_root: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineSummary:
    if output_root is None:
        output_root = Path.cwd()
    output_root = Path(output_root)

    result_root = output_root / "result"
    stage_directories = {1: result_root / "1", 2: result_root / "2"}
    for stage_dir in stage_directories.values():
        stage_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(path) for path in file_paths]
    identity_agent = IdentityGroupingAgent()
    grouped, skipped = identity_agent.analyze_images(
        paths,
        progress_callback=progress_callback,
        stage_name="Seleccionando imagenes",
    )

    crop_agent = PortraitCroppingAgent()

    batches: list[IdentityBatch] = []
    total_persons = len(grouped)
    progress_stage = "Recortando imagenes"
    if progress_callback:
        progress_callback(progress_stage, 0, total_persons)

    for index, (person_id, faces) in enumerate(grouped.items(), start=1):
        if progress_callback:
            progress_callback(progress_stage, index, total_persons)

        processed_faces: list[SelectedFace] = []

        # Take up to 2 photos of each person directly
        for rank, face in enumerate(faces[:2], start=1):
            image = cv2.imread(str(face.image_path))
            if image is None:
                continue

            crop = crop_agent.compute_crop(face)
            processed_faces.append(
                SelectedFace(
                    analysis=face,
                    crop=crop,
                )
            )

            stage_dir = stage_directories.get(rank)
            if stage_dir is None:
                continue

            destination = stage_dir / f"{person_id}{face.image_path.suffix}"
            # Crop and save with quality control
            crop_agent.crop_and_save(image, crop, destination)

        if processed_faces:
            batches.append(
                IdentityBatch(person_id=person_id, selected_faces=processed_faces)
            )

    return PipelineSummary(
        batches=batches,
        total_images=len(paths),
        skipped_files=skipped,
    )
