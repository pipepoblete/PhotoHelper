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
        "sharpness": 0.3,
        "eye_gaze": 0.2,
        "pose_quality": 0.1,
        "expression": 0.4,
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

            sharpness = self._sharpness(image, face.face_bbox)
            eye_gaze = self._eye_gaze_quality(face, image)
            pose_quality = self._pose_quality(face)
            expression = self._expression(face)
            eyes_visible = self._eyes_visible(face, image)
            has_glasses = self._has_glasses(face, image)
            features.append(
                (face, sharpness, eye_gaze, pose_quality, expression, eyes_visible, has_glasses)
            )

        if not features:
            return []

        sharpness_vals = [feat[1] for feat in features]
        eye_gaze_scores = [feat[2] for feat in features]
        pose_scores = [feat[3] for feat in features]
        expression_scores = [feat[4] for feat in features]
        eyes_visible_scores = [feat[5] for feat in features]

        norm_sharpness = self._normalize(sharpness_vals)
        norm_eye_gaze = self._normalize(eye_gaze_scores)
        norm_pose = self._normalize(pose_scores)
        norm_expression = self._normalize(expression_scores)
        norm_eyes_visible = self._normalize(eyes_visible_scores)

        scored: list[PhotoScore] = []
        for (face, _, _, _, _, _, has_glasses), s, gaze, pose, expression, eyes_vis in zip(
            features, norm_sharpness, norm_eye_gaze, norm_pose, norm_expression, norm_eyes_visible
        ):
            if has_glasses:
                normal_score = (
                    self.weights["sharpness"] * s
                    + self.weights["eye_gaze"] * gaze
                    + self.weights["pose_quality"] * pose
                    + self.weights["expression"] * expression
                )
                score = 0.5 * eyes_vis + 0.5 * normal_score
            else:
                score = (
                    self.weights["sharpness"] * s
                    + self.weights["eye_gaze"] * gaze
                    + self.weights["pose_quality"] * pose
                    + self.weights["expression"] * expression
                )
            scored.append(PhotoScore(face, score))

        scored.sort(key=lambda entry: entry.score, reverse=True)
        return [entry.result for entry in scored[: self.max_per_person]]

    @staticmethod
    def _sharpness(
        image: np.ndarray, face_bbox: tuple[int, int, int, int] | None = None
    ) -> float:
        """Calculate sharpness focusing on the face region when available."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            h, w = gray.shape
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            if x2 > x1 and y2 > y1:
                face_region = gray[y1:y2, x1:x2]
                return float(cv2.Laplacian(face_region, cv2.CV_64F).var())

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
    def _has_glasses(face: FaceAnalysisResult, image: np.ndarray) -> bool:
        """Detect if person is wearing glasses using edges near the eye region."""
        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        eye_y = min(left_eye[1], right_eye[1])
        eye_spacing = abs(right_eye[0] - left_eye[0])

        roi_y1 = max(0, int(eye_y - eye_spacing * 0.3))
        roi_y2 = min(h, int(eye_y + eye_spacing * 0.5))
        roi_x1 = max(0, int(min(left_eye[0], right_eye[0]) - eye_spacing * 0.3))
        roi_x2 = min(w, int(max(left_eye[0], right_eye[0]) + eye_spacing * 0.3))

        if roi_y2 <= roi_y1 or roi_x2 <= roi_x1:
            return False

        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=int(eye_spacing * 0.3),
            maxLineGap=10,
        )

        if lines is None:
            return False

        horizontal_count = 0
        roi_height = roi_y2 - roi_y1
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < roi_height * 0.1:
                line_length = abs(x2 - x1)
                if line_length > eye_spacing * 0.2:
                    horizontal_count += 1

        return horizontal_count >= 2

    @staticmethod
    def _eyes_visible(face: FaceAnalysisResult, image: np.ndarray) -> float:
        """Score based purely on reflected flash coverage on glasses.

        Returns 0.0-2.0 where 2.0 means no visible reflection (both eyes clear)
        and 0.0 means eyes are fully covered by bright reflections.
        """
        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        eye_spacing = abs(right_eye[0] - left_eye[0])
        eye_width = int(eye_spacing * 0.4)
        eye_height = int(eye_spacing * 0.3)

        left_x1 = max(0, int(left_eye[0] - eye_width))
        left_x2 = min(w, int(left_eye[0] + eye_width))
        left_y1 = max(0, int(left_eye[1] - eye_height))
        left_y2 = min(h, int(left_eye[1] + eye_height))

        right_x1 = max(0, int(right_eye[0] - eye_width))
        right_x2 = min(w, int(right_eye[0] + eye_width))
        right_y1 = max(0, int(right_eye[1] - eye_height))
        right_y2 = min(h, int(right_eye[1] + eye_height))

        if (
            left_x2 <= left_x1
            or left_y2 <= left_y1
            or right_x2 <= right_x1
            or right_y2 <= right_y1
        ):
            return 0.0

        left_eye_roi = gray[left_y1:left_y2, left_x1:left_x2]
        right_eye_roi = gray[right_y1:right_y2, right_x1:right_x2]

        left_reflection_threshold = max(220, np.percentile(left_eye_roi, 90))
        right_reflection_threshold = max(220, np.percentile(right_eye_roi, 90))

        left_reflection_pixels = np.sum(left_eye_roi > left_reflection_threshold)
        right_reflection_pixels = np.sum(right_eye_roi > right_reflection_threshold)

        left_total_pixels = left_eye_roi.size
        right_total_pixels = right_eye_roi.size

        left_reflection_coverage = left_reflection_pixels / max(1, left_total_pixels)
        right_reflection_coverage = right_reflection_pixels / max(1, right_total_pixels)

        # Score inversely with coverage; cap between 0 and 1 per eye
        left_score = max(0.0, 1.0 - left_reflection_coverage)
        right_score = max(0.0, 1.0 - right_reflection_coverage)

        # Sum both eyes: 0.0 (worst) to 2.0 (best)
        return left_score + right_score

    @staticmethod
    def _eye_gaze_quality(face: FaceAnalysisResult, image: np.ndarray | None = None) -> float:
        """Check if person is looking at camera based on eye symmetry."""
        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")
        nose = face.landmarks.get("nose")

        if not left_eye or not right_eye or not nose:
            return 0.0

        face_center_x = (face.face_bbox[0] + face.face_bbox[2]) / 2
        face_width = face.face_bbox[2] - face.face_bbox[0]

        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_offset = abs(eye_center_x - face_center_x)
        nose_offset = abs(nose[0] - face_center_x)

        eye_score = max(0.0, 1.0 - (eye_center_offset / (face_width * 0.5)))
        nose_score = max(0.0, 1.0 - (nose_offset / (face_width * 0.3)))

        return (eye_score + nose_score) / 2.0

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
