from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from agents.shared import FaceAnalysisResult


@dataclass(frozen=True)
class PhotoScore:
    result: FaceAnalysisResult
    score: float


class BestPhotoSelectionAgent:
    DEFAULT_WEIGHTS = {
        "sharpness": 0.6,
        "face_size": 0.0,
        "pose_quality": 0.3,
        "expression": 0.1,
    }

    def __init__(
        self, max_per_person: int = 2, weights: dict[str, float] | None = None
    ) -> None:
        self.max_per_person = max_per_person
        self.weights = weights or self.DEFAULT_WEIGHTS

    def select(self, faces: Sequence[FaceAnalysisResult]) -> list[FaceAnalysisResult]:
        if not faces:
            return []

        features = []
        for face in faces:
            image = cv2.imread(str(face.image_path))
            if image is None:
                continue

            sharpness = self._sharpness(image)
            face_size = face.face_area_ratio
            pose_quality = self._pose_quality(face)
            expression = self._expression(face)
            features.append((face, sharpness, face_size, pose_quality, expression))

        if not features:
            return []

        sharpness_vals = [feat[1] for feat in features]
        face_sizes = [feat[2] for feat in features]
        pose_scores = [feat[3] for feat in features]
        expression_scores = [feat[4] for feat in features]

        norm_sharpness = self._normalize(sharpness_vals)
        norm_face_size = self._normalize(face_sizes)
        norm_pose = self._normalize(pose_scores)
        norm_expression = self._normalize(expression_scores)

        scored: list[PhotoScore] = []
        for (face, _, _, _, _), s, size, pose, expression in zip(
            features, norm_sharpness, norm_face_size, norm_pose, norm_expression
        ):
            score = (
                self.weights["sharpness"] * s
                + self.weights["face_size"] * size
                + self.weights["pose_quality"] * pose
                + self.weights["expression"] * expression
            )
            scored.append(PhotoScore(face, score))

        scored.sort(key=lambda entry: entry.score, reverse=True)
        return [entry.result for entry in scored[: self.max_per_person]]

    @staticmethod
    def _sharpness(image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _pose_quality(face: FaceAnalysisResult) -> float:
        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")
        if not left_eye or not right_eye:
            return 0.0

        face_height = max(1, face.face_bbox[3] - face.face_bbox[1])
        vertical_misalignment = abs(left_eye[1] - right_eye[1])
        return max(0.0, 1.0 - vertical_misalignment / face_height)

    @staticmethod
    def _expression(face: FaceAnalysisResult) -> float:
        mouth_left = face.landmarks.get("mouth_left")
        mouth_right = face.landmarks.get("mouth_right")
        chin = face.landmarks.get("chin")
        if not mouth_left or not mouth_right or not chin:
            return 0.0

        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        face_height = max(1, face.face_bbox[3] - face.face_bbox[1])
        openness = max(0.0, chin[1] - mouth_center_y)
        return min(1.0, openness / (face_height * 0.4))

    @staticmethod
    def _normalize(values: Iterable[float]) -> list[float]:
        values = list(values)
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return [0.0 for _ in values]
        span = maximum - minimum
        return [(val - minimum) / span for val in values]
