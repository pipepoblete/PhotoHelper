from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.cluster import DBSCAN

from agents.shared import FaceAnalysisResult, ProgressCallback


class IdentityGroupingAgent:
    MODEL_NAME = "buffalo_l"

    def __init__(
        self,
        detection_size: tuple[int, int] = (640, 640),
        cluster_eps: float = 0.35,
        min_samples: int = 1,
        ctx_id: int = -1,
    ) -> None:
        self._cluster_eps = cluster_eps
        self._min_samples = min_samples
        self._analyzer = FaceAnalysis(name=self.MODEL_NAME)
        self._analyzer.prepare(ctx_id=ctx_id, det_size=detection_size)

    def analyze_images(
        self,
        image_paths: Iterable[Path],
        progress_callback: ProgressCallback | None = None,
        stage_name: str | None = None,
    ) -> tuple[dict[str, list[FaceAnalysisResult]], list[Path]]:
        results: list[FaceAnalysisResult] = []
        skipped: list[Path] = []
        
        image_paths_list = list(image_paths)
        total = len(image_paths_list)

        for idx, path in enumerate(image_paths_list, start=1):
            try:
                result = self._analyze_single(path)
            except Exception:  # pragma: no cover - best effort
                skipped.append(path)
                continue

            if result is None:
                skipped.append(path)
            else:
                results.append(result)
            
            if progress_callback and stage_name:
                progress_callback(stage_name, idx, total)

        if not results:
            return {}, skipped

        embeddings = np.stack(
            [np.asarray(r.embedding, dtype=np.float32) for r in results]
        )
        clustering = DBSCAN(
            metric="cosine", eps=self._cluster_eps, min_samples=self._min_samples
        ).fit(embeddings)

        label_map: dict[int, str] = {}
        grouped: dict[str, list[FaceAnalysisResult]] = {}
        for person_result, label in zip(results, clustering.labels_):
            group_key = self._label_to_key(label, label_map)
            grouped.setdefault(group_key, []).append(person_result)

        return grouped, skipped

    @staticmethod
    def _label_to_key(label: int, label_map: dict[int, str]) -> str:
        if label not in label_map:
            label_map[label] = f"person_{len(label_map)}"
        return label_map[label]

    def _analyze_single(self, image_path: Path) -> FaceAnalysisResult | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        faces = self._analyzer.get(image)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda detection: (detection.bbox[2] - detection.bbox[0])
            * (detection.bbox[3] - detection.bbox[1]),
        )
        bbox_array = face.bbox
        x1, y1, x2, y2 = map(int, bbox_array[:4])
        img_h, img_w = image.shape[:2]
        bbox = (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))

        landmarks = self._extract_landmarks(face, bbox)
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        face_ratio = area / float(img_w * img_h)

        embedding = np.asarray(face.embedding, dtype=np.float32).tolist()

        return FaceAnalysisResult(
            image_path=image_path,
            face_bbox=bbox,
            landmarks=landmarks,
            embedding=embedding,
            face_area_ratio=face_ratio,
            image_size=(img_w, img_h),
        )

    @staticmethod
    def _extract_landmarks(
        face, bbox: tuple[int, int, int, int]
    ) -> dict[str, tuple[float, float]]:
        landmarks: dict[str, tuple[float, float]] = {}
        if hasattr(face, "kps") and len(face.kps) >= 5:
            left_eye, right_eye, nose, mouth_left, mouth_right = face.kps[:5]
            landmarks.update(
                {
                    "left_eye": (float(left_eye[0]), float(left_eye[1])),
                    "right_eye": (float(right_eye[0]), float(right_eye[1])),
                    "nose": (float(nose[0]), float(nose[1])),
                    "mouth_left": (float(mouth_left[0]), float(mouth_left[1])),
                    "mouth_right": (float(mouth_right[0]), float(mouth_right[1])),
                }
            )

        centroid_x = (bbox[0] + bbox[2]) / 2
        landmarks["chin"] = (centroid_x, float(bbox[3]))
        return landmarks
