import os
import sys
from pathlib import Path

# Add paths for modules BEFORE any other imports
sys.path.insert(0, str(Path(__file__).parent.parent / "MAIN_MODULE" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "DEPARTMENT_CLASSIFICATION" / "train_model"))

from datetime import datetime
import sqlite3
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from ultralytics import YOLO
from application import ObjectDetector, VideoReader, SessionRepository
from domain import PriceTag, SessionStats
from predict_single import DepartmentPredictor
from crop_extraction import CropScorer


class YoloObjectDetector(ObjectDetector):
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45):
        config_path = Path(__file__).parent.parent / "params.yaml"
        config = OmegaConf.load(config_path)
        
        self.conf = config.main_extraction.conf_threshold
        self.iou = config.main_extraction.iou_threshold
        self.tracker_config = config.main_extraction.get("tracker_config", "bytetrack.yaml")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Критическая ошибка: Веса модели не найдены по пути: {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        results = self.model.track(
            source=frame,
            persist=False,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]
        
        boxes_data = []
        if results.boxes is not None and results.boxes.id is not None:
            for box, track_id, conf in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.id.cpu().numpy(),
                results.boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = box
                boxes_data.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(track_id)))
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return boxes_data
    
    def track(self, frame: np.ndarray, persist: bool = False) -> Any:
        """YOLO трекинг с опцией persist"""
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker_config,
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )
        return results[0]


class OpenCvVideoReader(VideoReader):
    def __init__(self):
        self.cap = None

    def open(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {path}")

    def get_total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 0

    def get_fps(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap else 0

    def read_frame(self, rotation: str) -> Optional[np.ndarray]:
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        if rotation == "90° против часовой":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "90° по часовой":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "180°":
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def close(self) -> None:
        if self.cap:
            self.cap.release()


class SqliteSessionRepository(SessionRepository):
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent / "backend_analytics.db")
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_filename TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    rotation TEXT,
                    total_frames INTEGER,
                    total_detections INTEGER,
                    elapsed_time REAL,
                    fps REAL,
                    mode_department TEXT,
                    mode_department_count INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    filename TEXT,
                    product_name TEXT,
                    price_default REAL,
                    price_card REAL,
                    price_discount TEXT,
                    barcode INTEGER,
                    discount_amount TEXT,
                    id_sku TEXT,
                    print_datetime TEXT,
                    code INTEGER,
                    additional_info TEXT,
                    color TEXT,
                    special_symbols TEXT,
                    frame_timestamp INTEGER,
                    x_min INTEGER,
                    y_min INTEGER,
                    x_max INTEGER,
                    y_max INTEGER,
                    qr_code_barcode INTEGER,
                    price1_qr REAL,
                    is_problematic INTEGER,
                    department TEXT,
                    department_prob REAL,
                    track_id INTEGER,
                    SYS_trash INTEGER,
                    SYS_quality_confidence REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
            """)
            conn.commit()

    def save_session(self, tag: str, stats: SessionStats, rows_crops: List[Dict], video_filename: str, rotation: str) -> int:
        """Сохранить сессию, вернуть ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute("""
                INSERT INTO sessions (
                    video_filename, tag, rotation,
                    total_frames, total_detections, elapsed_time, fps,
                    mode_department, mode_department_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_filename, tag, rotation,
                stats.total_frames, stats.total_detections, stats.elapsed_time, stats.fps,
                stats.mode_department, stats.mode_department_count, now
            ))
            session_id = cursor.lastrowid
            
            for row in rows_crops:
                cursor.execute("""
                    INSERT INTO price_tags (
                        session_id, filename, product_name, price_default, price_card, price_discount,
                        barcode, discount_amount, id_sku, print_datetime, code, additional_info,
                        color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
                        qr_code_barcode, price1_qr, is_problematic, department, department_prob, track_id,
                        SYS_trash, SYS_quality_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, row['filename'], row['product_name'], row['price_default'], row['price_card'], row['price_discount'],
                    row['barcode'], row['discount_amount'], row['id_sku'], row['print_datetime'], row['code'], row['additional_info'],
                    row['color'], row['special_symbols'], row['frame_timestamp'], row['x_min'], row['y_min'], row['x_max'], row['y_max'],
                    row['qr_code_barcode'], row['price1_qr'], row['is_problematic'], row['department'], row['department_prob'], row['track_id'],
                    1 if row.get('SYS_trash', False) else 0, row.get('SYS_quality_confidence', 0.0)
                ))
            conn.commit()
            return session_id

    def get_problematic_stats(self) -> Tuple[int, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM price_tags WHERE is_problematic = 0")
            normal = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM price_tags WHERE is_problematic = 1")
            problematic = cursor.fetchone()[0]
            return normal, problematic

    def get_daily_trends(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT created_at, SUM(total_detections) 
                FROM sessions 
                GROUP BY created_at 
                ORDER BY created_at ASC 
                LIMIT 7
            """)
            rows = cursor.fetchall()
            return [{'date': r[0], 'count': r[1]} for r in rows]
    
    def get_session_stats(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Получить статистику сессии по ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0], 'video_filename': row[1], 'tag': row[2], 'rotation': row[3],
                    'total_frames': row[4], 'total_detections': row[5], 'elapsed_time': row[6],
                    'fps': row[7], 'mode_department': row[8], 'mode_department_count': row[9],
                    'created_at': row[10]
                }
            return None
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Получить все сессии"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, video_filename, tag, rotation, total_frames, total_detections,
                       elapsed_time, fps, mode_department, mode_department_count, created_at
                FROM sessions ORDER BY created_at DESC
            """)
            return [
                {
                    'id': r[0], 'video_filename': r[1], 'tag': r[2], 'rotation': r[3],
                    'total_frames': r[4], 'total_detections': r[5], 'elapsed_time': r[6],
                    'fps': r[7], 'mode_department': r[8], 'mode_department_count': r[9],
                    'created_at': r[10]
                } for r in cursor.fetchall()
            ]
    
    def get_department_distribution(self) -> List[Dict[str, Any]]:
        """Распределение отделов по всем сессиям"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mode_department, COUNT(*) as session_count, SUM(total_detections) as total_detections
                FROM sessions
                GROUP BY mode_department
                ORDER BY session_count DESC
            """)
            return [
                {'department': r[0], 'session_count': r[1], 'total_detections': r[2]}
                for r in cursor.fetchall()
            ]
    
    def get_daily_good_bad_stats(self) -> List[Dict[str, Any]]:
        """Статистика хороших/плохих ценников по дням (для всех сессий)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DATE(s.created_at) as date, 
                       COUNT(CASE WHEN pt.SYS_trash = 0 THEN 1 END) as good,
                       COUNT(CASE WHEN pt.SYS_trash = 1 THEN 1 END) as bad
                FROM sessions s
                JOIN price_tags pt ON s.id = pt.session_id
                GROUP BY DATE(s.created_at)
                ORDER BY date ASC
            """)
            return [{'date': r[0], 'good': r[1], 'bad': r[2]} for r in cursor.fetchall()]
    
    def get_today_department_stats(self) -> List[Dict[str, Any]]:
        """Статистика по отделам (mode_department) за сегодня"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.mode_department as department,
                       COUNT(DISTINCT s.id) as sessions,
                       SUM(s.total_detections) as total,
                       SUM(CASE WHEN pt.SYS_trash = 0 THEN 1 ELSE 0 END) as good,
                       SUM(CASE WHEN pt.SYS_trash = 1 THEN 1 ELSE 0 END) as bad
                FROM sessions s
                LEFT JOIN price_tags pt ON s.id = pt.session_id
                WHERE DATE(s.created_at) = DATE('now')
                GROUP BY s.mode_department
                ORDER BY total DESC
            """)
            return [{'department': r[0], 'sessions': r[1], 'total': r[2], 'good': r[3], 'bad': r[4]} for r in cursor.fetchall()]
