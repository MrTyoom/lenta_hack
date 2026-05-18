import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import logging
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from ultralytics import YOLO

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@dataclass
class CropCandidate:
    score: float
    crop: np.ndarray
    frame_index: int
    confidence: float
    bbox: List[int]
    bbox_original: List[int] = None


class CropScorer:

    def __init__(
        self,
        min_width: int,
        min_height: int,
        sharpness_threshold: float,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.sharpness_threshold = sharpness_threshold

    def compute_score(
        self,
        crop: np.ndarray,
        confidence: float,
    ) -> Optional[float]:

        if crop.size == 0:
            return None

        height, width = crop.shape[:2]

        if width < self.min_width or height < self.min_height:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


        area = width * height
        aspect_ratio = width / (height + 1e-6)

        if not (0.9 <= aspect_ratio <= 8.0):
            return None

        
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness < self.sharpness_threshold:
            return None

        area_score = np.sqrt(area)

        score = (
            sharpness * 0.5
            + area_score * 0.4
            + confidence * 100 * 0.2
        )

        return float(score)

class VideoObjectDetector:
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    def __init__(self, config: OmegaConf) -> None:
        self.config = config

        logger.info("CUDA available: %s", torch.cuda.is_available())

        if torch.cuda.is_available():
            logger.info("GPU: %s", torch.cuda.get_device_name(0))

        self.model = YOLO(config.model_path).to(DEVICE)

        self.crop_scorer = CropScorer(
            min_width=config.min_crop_width,
            min_height=config.min_crop_height,
            sharpness_threshold=config.sharpness_threshold,
        )

        self.output_dir = Path(config.output_folder)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        start_time = time.time()

        video_files = self._get_video_files()

        if not video_files:
            logger.warning("No video files found")
            return

        logger.info("Found %d videos", len(video_files))

        for video_path in video_files:
            self.process_video(video_path)

        elapsed_minutes = (time.time() - start_time) / 60

        logger.info("Finished in %.2f minutes", elapsed_minutes)

    def process_video(self, video_path: Path) -> None:
        logger.info("Processing video: %s", video_path.name)

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            return

        best_crops: Dict[int, List[CropCandidate]] = defaultdict(list)

        frame_index = 0
        total_detections = 0

        try:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_index += 1

                frame = self._preprocess_frame(frame)

                detections = self._track_objects(frame)

                if detections is None:
                    continue

                boxes, confidences, track_ids = detections

                for box, confidence, track_id in zip(
                    boxes,
                    confidences,
                    track_ids,
                ):
                    crop_candidate = self._build_crop_candidate(
                        frame=frame,
                        box=box,
                        confidence=confidence,
                        frame_index=frame_index,
                    )

                    if crop_candidate is None:
                        continue

                    best_crops[track_id].append(crop_candidate)

                    best_crops[track_id] = sorted(
                        best_crops[track_id],
                        key=lambda item: item.score,
                        reverse=True,
                    )[: self.config.top_k]

                    total_detections += 1

                if frame_index % 100 == 0:
                    logger.info(
                        "Frame=%d | detections=%d",
                        frame_index,
                        total_detections,
                    )

        finally:
            cap.release()

        self._save_best_crops(
            video_name=video_path.stem,
            best_crops=best_crops,
        )

        logger.info("Finished video: %s", video_path.name)

    def _get_video_files(self) -> List[Path]:
        input_dir = Path(self.config.input_folder)

        return [
            path
            for path in input_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in self.VIDEO_EXTENSIONS
                and not path.name.startswith("~")
            )
        ]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.rotate_frames:
            frame = cv2.rotate(
                frame,
                cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        return frame

    def _track_objects(
        self,
        frame: np.ndarray,
    ) -> Optional[tuple]:

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.config.tracker_config,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            verbose=False,
        )

        result = results[0]

        if (
            result.boxes is None
            or result.boxes.id is None
        ):
            return None

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        return boxes, confidences, track_ids

    def _build_crop_candidate(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        confidence: float,
        frame_index: int,
    ) -> Optional[CropCandidate]:

        x1, y1, x2, y2 = map(int, box)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        crop = frame[y1:y2, x1:x2]

        score = self.crop_scorer.compute_score(
            crop=crop,
            confidence=confidence,
        )

        if score is None:
            return None

        return CropCandidate(
            score=score,
            crop=crop.copy(),
            frame_index=frame_index,
            confidence=float(confidence),
            bbox=[x1, y1, x2, y2],
        )

    def _save_best_crops(
        self,
        video_name: str,
        best_crops: Dict[int, List[CropCandidate]],
    ) -> None:

        output_dir = (
            self.output_dir
            / "best_crops"
            / video_name
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        for track_id, candidates in best_crops.items():

            track_dir = output_dir / f"id_{track_id}"
            track_dir.mkdir(parents=True, exist_ok=True)

            for index, item in enumerate(candidates, start=1):

                filename = (
                    f"top{index}"
                    f"_frame{item.frame_index}"
                    f"_conf{item.confidence:.2f}"
                    f"_score{item.score:.1f}.jpg"
                )

                save_path = track_dir / filename

                cv2.imwrite(
                    str(save_path),
                    item.crop,
                )