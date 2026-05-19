import os
import sys
from pathlib import Path

# Add paths for modules BEFORE any other imports
sys.path.insert(0, str(Path(__file__).parent.parent / "MAIN_MODULE" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "DEPARTMENT_CLASSIFICATION" / "train_model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "CROP_QUALITY_CLASSIFICATION"))
sys.path.insert(0, str(Path(__file__).parent.parent / "IMG PREPROCESSING"))

import time
import gc
import torch
import cv2
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from collections import defaultdict, Counter
import numpy as np
from omegaconf import OmegaConf
from domain import PriceTag, SessionStats, CropCandidate
from crop_extraction import CropScorer
from predict_single import DepartmentPredictor
from quality_classifier.predict import quality_classifier
from color_classifier import process_dataset


class ProgressListener(ABC):
    @abstractmethod
    def on_progress(self, processed: int, total: int, detections: int, stage: int = 1, stage_name: str = "Детекция"): pass


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]: pass
    
    @abstractmethod
    def track(self, frame: np.ndarray, persist: bool = False) -> Any: pass


class VideoReader(ABC):
    @abstractmethod
    def open(self, path: str) -> None: pass
    @abstractmethod
    def read_frame(self, rotation: str) -> Optional[np.ndarray]: pass
    @abstractmethod
    def get_total_frames(self) -> int: pass
    @abstractmethod
    def get_fps(self) -> int: pass
    @abstractmethod
    def close(self) -> None: pass


class SessionRepository(ABC):
    @abstractmethod
    def save_session(self, tag: str, stats: SessionStats, rows_crops: List[Dict], video_filename: str, rotation: str) -> int: pass
    @abstractmethod
    def get_problematic_stats(self) -> Tuple[int, int]: pass
    @abstractmethod
    def get_daily_trends(self) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def get_all_sessions(self) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def get_department_distribution(self) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def get_daily_good_bad_stats(self) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def get_today_department_stats(self) -> List[Dict[str, Any]]: pass


class DepartmentClassifier:
    def __init__(self, model_path: str, class_names_path: str):
        self.predictor = DepartmentPredictor(model_path, class_names_path)
    
    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        dept, prob = self.predictor.predict_np(frame)[0]
        return dept, prob


class ProcessVideoUseCase:
    def __init__(self, detector: ObjectDetector, video_reader: VideoReader, repository: SessionRepository):
        self.detector = detector
        self.video_reader = video_reader
        self.repository = repository
        
        config = OmegaConf.load(Path(__file__).parent.parent / "params.yaml")
        
        self.crop_scorer = CropScorer(
            min_width=config.main_extraction.min_crop_width,
            min_height=config.main_extraction.min_crop_height,
            sharpness_threshold=config.main_extraction.sharpness_threshold,
        )
        
        dept_model_path = Path(__file__).parent.parent / config.department_classifier.model_path
        class_names_path = Path(__file__).parent.parent / "DEPARTMENT_CLASSIFICATION/train_model/models/class_names.json"
        self.classifier = DepartmentClassifier(str(dept_model_path), str(class_names_path))
        
        quality_model_path = Path(__file__).parent.parent / config.quality_classifier.model_path
        self.quality_model_path = str(quality_model_path)
        
        self.config = config
        self.best = defaultdict(list)
        self.seg_preds = [[] for _ in range(5)]
        self.frame_to_seg = {}
        self.segment_frames = []
        self.video_fps = 30

    def execute(self, video_path: str, rotation: str, session_tag: str, progress_listener: Optional[ProgressListener] = None) -> Tuple[List[Dict], SessionStats]:
        config = self.config
        
        self.video_reader.open(video_path)
        total_frames = self.video_reader.get_total_frames()
        self.video_fps = self.video_reader.get_fps() if self.video_reader.get_fps() > 0 else 30
        
        # Разбить кадры на 5 сегментов
        all_frame_indices = list(range(total_frames))
        segment_size = max(len(all_frame_indices) // 5, 1)
        self.segment_frames = [
            all_frame_indices[i*segment_size:(i+1)*segment_size] if i < 4 
            else all_frame_indices[i*segment_size:]
            for i in range(5)
        ]
        
        # Создать mapping frame -> segment
        self.frame_to_seg = {}
        for sid, frames in enumerate(self.segment_frames):
            for f in frames:
                self.frame_to_seg[f] = sid
        
        # Сбросить коллекции
        self.best = defaultdict(list)
        self.seg_preds = [[] for _ in range(5)]
        
        fi = -1
        start_time = time.time()
        
        while True:
            frame = self.video_reader.read_frame(rotation)
            if frame is None:
                break
            
            fi += 1
            
            # YOLO track с persist=True
            res = self.detector.track(frame, persist=True)
            
            # Сбор кропов
            if res.boxes is not None and res.boxes.id is not None:
                for box, conf, tid in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.id.cpu().numpy().astype(int)
                ):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y1:y2, x1:x2]
                    
                    score = self.crop_scorer.compute_score(crop, conf)
                    if score is None:
                        continue
                    
                    c = CropCandidate(score, crop.copy(), fi, float(conf), [x1, y1, x2, y2])
                    self.best[tid].append(c)
                    self.best[tid].sort(key=lambda x: x.score, reverse=True)
                    self.best[tid] = self.best[tid][:config.main_extraction.top_k]
            
            # Department только для сегментов
            if fi in self.frame_to_seg:
                sid = self.frame_to_seg[fi]
                dept, prob = self.classifier.predict(frame)
                self.seg_preds[sid].append({
                    'frame': fi,
                    'time': fi / self.video_fps,
                    'department': dept,
                    'prob': prob
                })
            
            # Прогресс
            if progress_listener and (fi + 1) % 100 == 0:
                progress_listener.on_progress(fi + 1, total_frames, len(self.best), 1, "Детекция ценников")
        
        self.video_reader.close()
        
        # Этап 2: Классификация качества
        crops_for_quality = [(track_id, candidates[0].crop) 
                             for track_id, candidates in self.best.items() 
                             if candidates]
        
        if progress_listener:
            progress_listener.on_progress(len(crops_for_quality), len(crops_for_quality), len(self.best), 2, "Оценка качества")
        
        trash_map, confidence_map = quality_classifier(self.quality_model_path, crops_for_quality)
        
        # Этап 3: Определение цветов
        if progress_listener:
            progress_listener.on_progress(len(crops_for_quality), len(crops_for_quality), len(self.best), 3, "Определение цветов")
        
        color_results = process_dataset(crops_for_quality)
        
        # Этап 4: Классификация отделов (уже сделана во время детекции, но показываем как завершённый)
        if progress_listener:
            progress_listener.on_progress(len(self.seg_preds), len(self.seg_preds), len(self.best), 4, "Классификация отделов")
        
        # Собрать результаты
        rows_crops = []
        for track_id, candidates in self.best.items():
            for rank, c in enumerate(candidates):
                # Выбрать department для этого трека
                track_frames = [cand.frame_index for cand in candidates]
                track_depts = []
                for tf in track_frames:
                    if tf in self.frame_to_seg:
                        sid = self.frame_to_seg[tf]
                        for p in self.seg_preds[sid]:
                            if p['frame'] == tf:
                                track_depts.append(p['department'])
                
                dept_mode = Counter(track_depts).most_common(1)[0][0] if track_depts else "Не определён"
                dept_prob = (Counter(track_depts).most_common(1)[0][1] / len(track_depts) * 100) if track_depts else 0.0
                
                is_trash = trash_map.get(track_id, False)
                quality_conf = confidence_map.get(track_id, 0.0)
                crop_color = color_results.get(track_id, "white")
                
                rows_crops.append({
                    'filename': os.path.basename(video_path),
                    'SYS_track_id': track_id,
                    'SYS_rank': rank + 1,
                    'SYS_score': round(c.score, 1),
                    'SYS_confidence': round(c.confidence, 3),
                    'SYS_trash': is_trash,
                    'SYS_quality_confidence': round(quality_conf, 1),
                    'product_name': "Товар (OCR будет позже)",
                    'price_default': 0.0,
                    'price_card': 0.0,
                    'price_discount': "нет",
                    'barcode': 0,
                    'discount_amount': "нет",
                    'id_sku': "нет",
                    'print_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'code': 1,
                    'additional_info': f"conf={c.confidence:.2f}",
                    'color': crop_color,
                    'special_symbols': "нет",
                    'frame_timestamp': int(c.frame_index / self.video_fps * 1000),
                    'x_min': c.bbox[0],
                    'y_min': c.bbox[1],
                    'x_max': c.bbox[2],
                    'y_max': c.bbox[3],
                    'qr_code_barcode': 0,
                    'price1_qr': 0.0,
                    'is_problematic': 1 if is_trash else 0,
                    'department': dept_mode,
                    'department_prob': dept_prob,
                    'track_id': track_id,
                    'crop_image': c.crop,
                })
        
        # Вычислить mode department для всего видео
        all_depts = [p['department'] for seg in self.seg_preds for p in seg]
        mode_dept, mode_count = Counter(all_depts).most_common(1)[0] if all_depts else ("", 0)
        
        # Статистика
        elapsed_time = time.time() - start_time
        stats = SessionStats(
            total_frames=fi + 1,
            total_detections=len(rows_crops),
            elapsed_time=round(elapsed_time, 2),
            fps=round((fi + 1) / elapsed_time, 1) if elapsed_time > 0 else 0.0,
            mode_department=mode_dept,
            mode_department_count=mode_count
        )
        
        # Сохранить в БД
        self.repository.save_session(session_tag, stats, rows_crops, os.path.basename(video_path), rotation)
        
        # GPU очистка
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return rows_crops, stats


class GetAnalyticsUseCase:
    def __init__(self, repository: SessionRepository):
        self.repository = repository

    def get_dashboard_data(self) -> Dict[str, Any]:
        problem_stats = self.repository.get_problematic_stats()
        daily_trends = self.repository.get_daily_trends()
        daily_good_bad = self.repository.get_daily_good_bad_stats()
        today_dept = self.repository.get_today_department_stats()
        return {
            'problem_stats': problem_stats,
            'daily_trends': daily_trends,
            'daily_good_bad': daily_good_bad,
            'today_department': today_dept
        }
