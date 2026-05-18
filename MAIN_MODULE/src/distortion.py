from __future__ import annotations

import math
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class CameraSettings:
    imageSize: tuple
    diagonalMm: float
    focalLenMm: float


class DistortionCorrector:
    def __init__(self, cameraSettings: CameraSettings, distortionCoeffs: list, alpha=1.0):
        self.width = cameraSettings.imageSize[0]
        self.height = cameraSettings.imageSize[1]
        self.diagonal_mm = cameraSettings.diagonalMm
        self.focal_length_mm = cameraSettings.focalLenMm
        self.dist = np.array(distortionCoeffs, dtype=np.float32)
        self.K = self._calculate_camera_matrix()
        # alpha=1.0 сохраняет все пиксели (могут быть чёрные области)
        # alpha=0.0 обрезает до ROI (нет чёрных областей, но меньше поле зрения)
        self.new_cam_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
            self.K, self.dist, (self.width, self.height), alpha, (self.width, self.height)
        )
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = self.roi
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.new_cam_matrix,
            (self.width, self.height), cv2.CV_32FC1
        )

    def _calculate_camera_matrix(self) -> np.ndarray:
        aspect_ratio = self.width / self.height
        height_mm = self.diagonal_mm / math.sqrt(aspect_ratio ** 2 + 1)
        width_mm = aspect_ratio * height_mm
        fx = (self.focal_length_mm * self.width) / width_mm
        fy = (self.focal_length_mm * self.height) / height_mm
        return np.array([
            [fx, 0, self.width / 2],
            [0, fy, self.height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    def undistort_frame(self, frame: np.ndarray, crop_to_roi=False) -> np.ndarray:
        """
        frame: исходное изображение
        crop_to_roi: если True, обрезать до ROI (нет чёрных областей)
                     если False, сохранить весь фрейм (могут быть чёрные области)
        """
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        if crop_to_roi:
            return undistorted[self.roi_y:self.roi_y + self.roi_h,
                               self.roi_x:self.roi_x + self.roi_w]
        return undistorted

    def undistort_bbox(self, bbox: tuple) -> tuple:
        x_min, y_min, x_max, y_max = bbox
        pts = np.array([[[x_min, y_min], [x_max, y_min],
                         [x_max, y_max], [x_min, y_max]]], dtype=np.float32)
        corrected = cv2.undistortPoints(pts, self.K, self.dist, P=self.new_cam_matrix)
        corrected = corrected[0]
        new_x_min = int(corrected[:, 0].min() - self.roi_x)
        new_y_min = int(corrected[:, 1].min() - self.roi_y)
        new_x_max = int(corrected[:, 0].max() - self.roi_x)
        new_y_max = int(corrected[:, 1].max() - self.roi_y)
        return new_x_min, new_y_min, new_x_max, new_y_max


CAM_SETTINGS = CameraSettings(
    (3840, 2160),
    16.0 / 2.8,
    2.8
)

CAM_DISTORT_COEFFS = [
    -0.276, 0.06, 0.0084, -0.0016, -0.0044
]
