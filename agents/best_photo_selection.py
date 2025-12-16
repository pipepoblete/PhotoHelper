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
        "sharpness": 0.4,
        "eye_gaze": 0.2,
        "squint": 0.3,
        "expression": 0.1,
    }
    GLASSES_COLOR_VARIATION_THRESHOLD = (
        220  # Increase/decrease after trial-and-error runs
    )

    def __init__(
        self, max_per_person: int = 2, weights: dict[str, float] | None = None
    ) -> None:
        self.max_per_person = max_per_person
        merged = self.DEFAULT_WEIGHTS.copy()
        if weights:
            custom = dict(weights)
            if "squint" not in custom and "pose_quality" in custom:
                custom["squint"] = custom["pose_quality"]
            merged.update(custom)
        self.weights = merged

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
            squint_score = self._squint_score(face, image)
            expression = self._expression(face)
            eyes_visible = self._eyes_visible(face, image)
            has_glasses, glasses_variation = self._has_glasses(face, image)
            features.append(
                (
                    face,
                    sharpness,
                    eye_gaze,
                    squint_score,
                    expression,
                    eyes_visible,
                    has_glasses,
                    glasses_variation,
                )
            )

        if not features:
            return []

        # Promote group-level glasses detection: if any image shows glasses, treat all as glasses
        has_glasses_group = any(feat[6] for feat in features)

        sharpness_vals = [feat[1] for feat in features]
        eye_gaze_scores = [feat[2] for feat in features]
        squint_scores = [feat[3] for feat in features]
        expression_scores = [feat[4] for feat in features]
        eyes_visible_scores = [feat[5] for feat in features]

        norm_sharpness = self._normalize(sharpness_vals)
        norm_eye_gaze = self._normalize(eye_gaze_scores)
        norm_squint = self._normalize(squint_scores)
        norm_expression = self._normalize(expression_scores)
        norm_eyes_visible = self._normalize(eyes_visible_scores)

        scored: list[PhotoScore] = []
        for (face, _, _, _, _, _, _, _), s, gaze, squint, expression, eyes_vis in zip(
            features,
            norm_sharpness,
            norm_eye_gaze,
            norm_squint,
            norm_expression,
            norm_eyes_visible,
        ):
            if has_glasses_group:
                normal_score = (
                    self.weights["sharpness"] * s
                    + self.weights["eye_gaze"] * gaze
                    + self.weights["squint"] * squint
                    + self.weights["expression"] * expression
                )
                score = 0.5 * eyes_vis + 0.5 * normal_score
                print(score, eyes_vis, normal_score, s, squint, gaze, expression)
            else:
                score = (
                    self.weights["sharpness"] * s
                    + self.weights["eye_gaze"] * gaze
                    + self.weights["squint"] * squint
                    + self.weights["expression"] * expression
                )
                print(score, s, squint, gaze, expression)
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
    def _squint_score(face: FaceAnalysisResult, image: np.ndarray) -> float:
        """Estimate how open the eyes are; lower when squinting."""

        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")
        if not left_eye or not right_eye:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        eye_spacing = max(1.0, abs(right_eye[0] - left_eye[0]))
        half_width = int(max(4, eye_spacing * 0.35))
        half_height = int(max(3, eye_spacing * 0.25))

        def eye_open_fraction(center: tuple[float, float]) -> float:
            cx, cy = center
            x1 = max(0, int(cx - half_width))
            x2 = min(w, int(cx + half_width))
            y1 = max(0, int(cy - half_height))
            y2 = min(h, int(cy + half_height))
            if x2 <= x1 or y2 <= y1:
                return 0.0

            roi = gray[y1:y2, x1:x2]
            dynamic_threshold = max(40, np.percentile(roi, 35))
            dark_ratio = np.sum(roi < dynamic_threshold) / max(1, roi.size)
            # Map dark pixel coverage into 0-1 range. Higher coverage -> eyes more open.
            return float(np.clip((dark_ratio - 0.03) / 0.22, 0.0, 1.0))

        left_score = eye_open_fraction(left_eye)
        right_score = eye_open_fraction(right_eye)
        return (left_score + right_score) * 0.5

    def _has_glasses(
        self, face: FaceAnalysisResult, image: np.ndarray
    ) -> tuple[bool, float]:
        """Detect glasses by measuring color shifts along vertical line between eyes."""

        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")
        nose = face.landmarks.get("nose")

        if not left_eye or not right_eye:
            return False, 0.0

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        h, w, _ = lab.shape
        eye_spacing = max(1.0, abs(right_eye[0] - left_eye[0]))

        def point_color(cx: float, cy: float) -> np.ndarray | None:
            x = int(np.clip(cx, 0, w - 1))
            y = int(np.clip(cy, 0, h - 1))
            if x < 0 or x >= w or y < 0 or y >= h:
                return None
            return lab[y, x]

        def sample_vertical_line(
            x: float,
            y_center: float,
            height: float,
            steps: int = 12,
        ) -> list[np.ndarray]:
            samples: list[np.ndarray] = []
            y_start = y_center - height / 2
            y_end = y_center + height / 2
            for idx in range(steps):
                ratio = idx / max(steps - 1, 1)
                y = y_start + (y_end - y_start) * ratio
                color = point_color(x, y)
                if color is not None:
                    samples.append(color)
            return samples

        # Use middle of eyes or nose as x-coordinate
        eyes_center_x = (left_eye[0] + right_eye[0]) / 2
        x_pos = nose[0] if nose else eyes_center_x

        # Use eye level as y-center, one eye height up and down
        y_center = (left_eye[1] + right_eye[1]) / 2
        height = eye_spacing * 0.6  # Approximate eye height

        colors = sample_vertical_line(x_pos, y_center, height)
        if len(colors) < 2:
            return False, 0.0

        brightness = np.array([color[0] for color in colors])
        min_idx = int(np.argmin(brightness))
        max_idx = int(np.argmax(brightness))
        color_delta = float(np.linalg.norm(colors[max_idx] - colors[min_idx]))

        variation = color_delta
        has_glasses = variation <= self.GLASSES_COLOR_VARIATION_THRESHOLD
        print(has_glasses, variation)
        return has_glasses, variation

    @staticmethod
    def _eyes_visible(face: FaceAnalysisResult, image: np.ndarray) -> float:
        """Score visibility based on distance between pupils and brightest facial spot."""

        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")

        if not left_eye or not right_eye:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        eye_spacing = max(1.0, abs(right_eye[0] - left_eye[0]))
        half_width = int(max(4, eye_spacing * 0.4))
        half_height = int(max(3, eye_spacing * 0.3))

        x1 = max(0, int(face.face_bbox[0]))
        y1 = max(0, int(face.face_bbox[1]))
        x2 = min(w, int(face.face_bbox[2]))
        y2 = min(h, int(face.face_bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return 0.0

        face_roi = gray[y1:y2, x1:x2]
        _, max_val, _, max_loc = cv2.minMaxLoc(face_roi)
        highlight_threshold = max(200, np.percentile(face_roi, 98))
        if max_val < highlight_threshold:
            return 1.0

        white_point = (x1 + max_loc[0], y1 + max_loc[1])
        max_distance = (
            float(
                np.hypot(
                    face.face_bbox[2] - face.face_bbox[0],
                    face.face_bbox[3] - face.face_bbox[1],
                )
            )
            + 1e-6
        )

        pupil_points: list[tuple[float, float]] = []
        for eye_center in (left_eye, right_eye):
            _, pupil_point = BestPhotoSelectionAgent._pupil_orientation_and_position(
                gray, eye_center, half_width, half_height
            )
            if pupil_point is not None:
                pupil_points.append(pupil_point)

        if not pupil_points:
            return 0.0

        scores = []
        for pupil_point in pupil_points:
            distance = float(
                np.hypot(
                    pupil_point[0] - white_point[0], pupil_point[1] - white_point[1]
                )
            )
            scores.append(min(1.0, distance / max_distance))

        return float(np.mean(scores))

    @staticmethod
    def _pupil_orientation_and_position(
        gray: np.ndarray,
        center: tuple[float, float],
        half_width: int,
        half_height: int,
    ) -> tuple[float | None, tuple[float, float] | None]:
        h, w = gray.shape
        cx, cy = center
        x1 = max(0, int(cx - half_width))
        x2 = min(w, int(cx + half_width))
        y1 = max(0, int(cy - half_height))
        y2 = min(h, int(cy + half_height))
        if x2 <= x1 or y2 <= y1:
            return None, None

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None

        min_val, _, min_loc, _ = cv2.minMaxLoc(roi)
        if min_val == 0 and np.count_nonzero(roi) == 0:
            return None, None

        roi_center_x = roi.shape[1] / 2.0
        offset = (min_loc[0] - roi_center_x) / max(roi.shape[1] / 2.0, 1e-6)
        orientation = float(np.clip(offset, -1.0, 1.0))
        pupil_point = (float(x1 + min_loc[0]), float(y1 + min_loc[1]))
        return orientation, pupil_point

    @staticmethod
    def _eye_gaze_quality(
        face: FaceAnalysisResult, image: np.ndarray | None = None
    ) -> float:
        """Estimate gaze by combining head (nose) and pupil orientation."""

        left_eye = face.landmarks.get("left_eye")
        right_eye = face.landmarks.get("right_eye")
        nose = face.landmarks.get("nose")

        if image is None or not left_eye or not right_eye or not nose:
            return 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_left, _, face_right, _ = face.face_bbox
        face_center_x = (face_left + face_right) / 2.0
        face_half_width = max(1.0, (face_right - face_left) / 2.0)
        head_orientation = float(
            np.clip((nose[0] - face_center_x) / face_half_width, -1.0, 1.0)
        )

        eye_spacing = max(1.0, abs(right_eye[0] - left_eye[0]))
        half_width = int(max(4, eye_spacing * 0.35))
        half_height = int(max(3, eye_spacing * 0.25))

        eye_offsets = []
        for eye_center in (left_eye, right_eye):
            orientation, _ = BestPhotoSelectionAgent._pupil_orientation_and_position(
                gray, eye_center, half_width, half_height
            )
            if orientation is not None:
                eye_offsets.append(orientation)

        eye_orientation = float(np.mean(eye_offsets)) if eye_offsets else 0.0

        deviation = min(1.0, abs(head_orientation) + abs(eye_orientation))
        return max(0.0, 1.0 - deviation)

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
