from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from agents.shared import FaceAnalysisResult, PortraitCropResult


class PortraitCroppingAgent:
    """Produce 13 × 18 cm portraits using face-based margins."""

    CM_TO_INCH = 1 / 2.54
    MIN_FILE_SIZE_KB = 1000
    MAX_FILE_SIZE_KB = 1500

    def __init__(
        self,
        target_width_cm: float = 13.0,
        target_height_cm: float = 18.0,
        dpi: int = 300,
    ) -> None:
        self.target_width_cm = target_width_cm
        self.target_height_cm = target_height_cm
        self.dpi = dpi
        self.height_to_width_ratio = target_height_cm / target_width_cm
        self.target_size = (
            int(round(target_width_cm * self.CM_TO_INCH * dpi)),
            int(round(target_height_cm * self.CM_TO_INCH * dpi)),
        )

    def compute_crop(self, analysis: FaceAnalysisResult) -> PortraitCropResult:
        """Compute crop box using face dimensions as reference."""
        image_width, image_height = analysis.image_size
        left, top, right, bottom = analysis.face_bbox
        face_width = max(1, right - left)
        face_height = max(1, bottom - top)

        # Top margin: half of face height above the top of the head
        top_margin = face_height * 0.7
        upper_bound = max(0, int(round(top - top_margin)))

        # Horizontal margins: face width on each side
        left_margin = face_width * 1.2
        right_margin = face_width * 1.2

        # Calculate desired width based on face + margins
        desired_width = face_width + left_margin + right_margin
        desired_width = min(desired_width, image_width)

        # Calculate desired height to maintain 13cm × 18cm aspect ratio
        desired_height = int(round(desired_width * self.height_to_width_ratio))
        desired_height = min(desired_height, image_height - upper_bound)

        # Recalculate width if height constraint is tighter
        if desired_height < desired_width * self.height_to_width_ratio:
            desired_width = int(round(desired_height / self.height_to_width_ratio))
            desired_width = min(desired_width, image_width)

        # Ensure width is at least face width and convert to int
        width = int(round(max(desired_width, face_width)))
        width = min(width, image_width)

        # Recalculate height based on final width
        height = int(round(width * self.height_to_width_ratio))
        height = min(height, image_height - upper_bound)
        height = max(height, 1)

        # Calculate lower bound
        lower_bound = int(upper_bound + height)
        if lower_bound > image_height:
            lower_bound = int(image_height)
            height = lower_bound - upper_bound
            # Recalculate width if height was constrained
            width = int(round(height / self.height_to_width_ratio))
            width = min(width, image_width)

        # Center horizontally on face
        face_center_x = (left + right) / 2
        left_bound = int(round(face_center_x - width / 2))
        right_bound = int(left_bound + width)

        # Clamp to image boundaries
        if left_bound < 0:
            left_bound = 0
            right_bound = int(width)
        elif right_bound > image_width:
            right_bound = int(image_width)
            left_bound = int(image_width - width)

        # Ensure all values are integers
        crop_box = (
            int(left_bound),
            int(upper_bound),
            int(right_bound),
            int(lower_bound),
        )
        return PortraitCropResult(crop_box=crop_box, target_size=self.target_size)

    def crop_and_save(
        self,
        image: np.ndarray,
        crop_result: PortraitCropResult,
        output_path: Path,
    ) -> None:
        """Crop image, resize to target size, and save with quality control."""
        left, top, right, bottom = crop_result.crop_box
        # Ensure all indices are integers
        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)
        # Validate bounds
        if right <= left or bottom <= top:
            raise ValueError(
                f"Invalid crop box: left={left}, top={top}, right={right}, bottom={bottom}"
            )
        if left < 0 or top < 0 or right > image.shape[1] or bottom > image.shape[0]:
            raise ValueError(
                f"Crop box out of bounds: image shape={image.shape}, "
                f"crop_box=({left}, {top}, {right}, {bottom})"
            )
        cropped = image[top:bottom, left:right]

        # Resize to target size
        target_width, target_height = crop_result.target_size
        resized = cv2.resize(
            cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
        )

        # Determine output format
        output_path = Path(output_path)
        is_jpeg = output_path.suffix.lower() in {".jpg", ".jpeg"}

        if is_jpeg:
            # JPEG: adjust quality to meet file size requirements
            quality = 95
            min_quality = 50
            max_quality = 100
            max_iterations = 20
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, encoded = cv2.imencode(".jpg", resized, encode_params)
                if not success:
                    # Fallback to default quality
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
                    _, encoded = cv2.imencode(".jpg", resized, encode_params)
                    break

                file_size_kb = len(encoded) / 1024

                if self.MIN_FILE_SIZE_KB <= file_size_kb <= self.MAX_FILE_SIZE_KB:
                    break

                if file_size_kb < self.MIN_FILE_SIZE_KB:
                    # Increase quality
                    if quality >= max_quality:
                        break
                    quality = min(max_quality, quality + 5)
                else:
                    # Decrease quality
                    if quality <= min_quality:
                        break
                    quality = max(min_quality, quality - 5)

            output_path.write_bytes(encoded.tobytes())
        else:
            # PNG or other formats: save with default compression
            cv2.imwrite(str(output_path), resized)
