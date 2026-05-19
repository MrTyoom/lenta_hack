import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "MAIN_MODULE" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "DEPARTMENT_CLASSIFICATION" / "train_model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "CROP_QUALITY_CLASSIFICATION"))
sys.path.insert(0, str(Path(__file__).parent.parent / "IMG PREPROCESSING"))
sys.path.insert(0, str(Path(__file__).parent.parent / "VLM_MODULE"))
sys.path.insert(0, str(Path(__file__).parent.parent / "LLMTEXT"))
sys.path.insert(0, str(Path(__file__).parent.parent / "VRAM_CLEAN"))

import time
import gc
import torch
import cv2
import pandas as pd
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
from detect import load_vlm_model, vlm_predict_crops
from hf_sku_matcher import HFSKUMatcher, clean_ocr_text
from product_matcher import find_top5_matches
from vram_cleanup import cleanup_vram


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
        self.is_test = config.get('is_test', False)
        self.vlm_model_path = str(Path(__file__).parent.parent / "VLM_MODULE" / "AVITO")
        self.vlm_config_name = "High Quality 8-bit Pro"
        self.best = defaultdict(list)
        self.seg_preds = [[] for _ in range(5)]
        self.frame_to_seg = {}
        self.segment_frames = []
        self.video_fps = 30

    def _build_crops_df_for_vlm(self, rows_crops):
        rows_for_vlm = [r for r in rows_crops if not r.get('SYS_trash', False)]
        if not rows_for_vlm:
            return pd.DataFrame()
        df = pd.DataFrame(rows_for_vlm)
        df = df.rename(columns={
            'x_min_orig': 'x_min', 'y_min_orig': 'y_min',
            'x_max_orig': 'x_max', 'y_max_orig': 'y_max'
        })
        df['crop_array'] = df['crop_image'].apply(lambda x: x if x is not None else None)
        df = df.drop(columns=['crop_image'], errors='ignore')
        return df

    def _merge_vlm_results(self, rows_crops, df_final):
        vlm_lookup = {}
        for _, row in df_final.iterrows():
            tid = row.get('SYS_track_id')
            if tid is not None:
                vlm_lookup[tid] = row.to_dict()
        
        for row in rows_crops:
            tid = row.get('SYS_track_id')
            if tid in vlm_lookup:
                v = vlm_lookup[tid]
                row['product_name'] = str(v.get('product_name', '')) or row.get('product_name', '')
                price_default_raw = v.get('price_without_card', '')
                try:
                    row['price_default'] = float(price_default_raw) if price_default_raw and str(price_default_raw).strip() else 0.0
                except (ValueError, TypeError):
                    row['price_default'] = 0.0
                price_card_raw = v.get('price_with_card', '')
                try:
                    row['price_card'] = float(price_card_raw) if price_card_raw and str(price_card_raw).strip() else 0.0
                except (ValueError, TypeError):
                    row['price_card'] = 0.0
                row['price_discount'] = str(v.get('promo_price', '')) if v.get('promo_price', '') else row.get('price_discount', 'нет')
                row['barcode'] = v.get('barcode', row.get('barcode', 0))
                row['discount_amount'] = str(v.get('discount_size', '')) if v.get('discount_size', '') else row.get('discount_amount', 'нет')
                row['id_sku'] = str(v.get('id_sku', '')) if v.get('id_sku', '') and str(v.get('id_sku', '')) != 'None' else 'нет'
                row['print_datetime'] = str(v.get('print_date', '')) if v.get('print_date', '') else row.get('print_datetime', '')
                row['special_symbols'] = str(v.get('layout_code', '')) if v.get('layout_code', '') else row.get('special_symbols', 'нет')
                row['code'] = v.get('article', row.get('code', 1))
        return rows_crops

    def execute(self, video_path: str, rotation: str, session_tag: str, progress_listener: Optional[ProgressListener] = None) -> Tuple[List[Dict], SessionStats]:
        config = self.config
        
        cleanup_vram(verbose=True)
        
        self.video_reader.open(video_path)
        total_frames = self.video_reader.get_total_frames()
        self.video_fps = self.video_reader.get_fps() if self.video_reader.get_fps() > 0 else 30
        
        all_frame_indices = list(range(total_frames))
        segment_size = max(len(all_frame_indices) // 5, 1)
        self.segment_frames = [
            all_frame_indices[i*segment_size:(i+1)*segment_size] if i < 4 
            else all_frame_indices[i*segment_size:]
            for i in range(5)
        ]
        
        self.frame_to_seg = {}
        for sid, frames in enumerate(self.segment_frames):
            for f in frames:
                self.frame_to_seg[f] = sid
        
        self.best = defaultdict(list)
        self.seg_preds = [[] for _ in range(5)]
        
        fi = -1
        start_time = time.time()
        
        while True:
            frame = self.video_reader.read_frame(rotation)
            if frame is None:
                break
            
            fi += 1
            
            res = self.detector.track(frame, persist=True)
            
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
            
            if fi in self.frame_to_seg:
                sid = self.frame_to_seg[fi]
                dept, prob = self.classifier.predict(frame)
                self.seg_preds[sid].append({
                    'frame': fi,
                    'time': fi / self.video_fps,
                    'department': dept,
                    'prob': prob
                })
            
            if progress_listener and (fi + 1) % 100 == 0:
                progress_listener.on_progress(fi + 1, total_frames, len(self.best), 1, "Детекция ценников")
            
            if self.is_test and len(self.best) >= 1:
                break
        
        self.video_reader.close()
        
        if self.is_test:
            first_tid = list(self.best.keys())[0]
            self.best = {first_tid: self.best[first_tid]}
        
        crops_for_quality = [(track_id, candidates[0].crop) 
                             for track_id, candidates in self.best.items() 
                             if candidates]
        
        if progress_listener:
            progress_listener.on_progress(len(crops_for_quality), len(crops_for_quality), len(self.best), 2, "Оценка качества")
        
        trash_map, confidence_map = quality_classifier(self.quality_model_path, crops_for_quality)
        
        if progress_listener:
            progress_listener.on_progress(len(crops_for_quality), len(crops_for_quality), len(self.best), 3, "Определение цветов")
        
        color_results = process_dataset(crops_for_quality)
        
        if progress_listener:
            progress_listener.on_progress(len(self.seg_preds), len(self.seg_preds), len(self.best), 4, "Классификация отделов")
        
        rows_crops = []
        for track_id, candidates in self.best.items():
            for rank, c in enumerate(candidates):
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
        
        # --- Этап 5: VLM OCR ---
        non_trash_rows = [r for r in rows_crops if not r.get('SYS_trash', False)]
        
        if non_trash_rows:
            if progress_listener:
                progress_listener.on_progress(0, len(non_trash_rows), len(self.best), 5, "VLM OCR распознавание")
            
            vlm_input_df = pd.DataFrame(non_trash_rows)
            vlm_input_df['crop_array'] = vlm_input_df['crop_image'].apply(lambda x: x if x is not None else None)
            cols_to_keep = ['SYS_track_id', 'SYS_rank', 'SYS_score', 'SYS_confidence', 'SYS_trash',
                           'x_min', 'y_min', 'x_max', 'y_max', 'crop_array']
            vlm_input_df = vlm_input_df[[c for c in cols_to_keep if c in vlm_input_df.columns]].copy()
            
            try:
                vlm_model, vlm_processor = load_vlm_model(self.vlm_model_path, config_name=self.vlm_config_name)
                df_vlm = vlm_predict_crops(vlm_input_df, vlm_model, vlm_processor, config_name=self.vlm_config_name)
                
                if progress_listener:
                    progress_listener.on_progress(len(non_trash_rows), len(non_trash_rows), len(self.best), 5, "VLM OCR распознавание")
            except Exception as e:
                print(f"[VLM] Ошибка: {e}")
                df_vlm = None
            finally:
                if 'vlm_model' in dir():
                    del vlm_model
                if 'vlm_processor' in dir():
                    del vlm_processor
                cleanup_vram()
            
            # --- Этап 6: Поиск товаров (product matching) ---
            if df_vlm is not None and not df_vlm.empty:
                if progress_listener:
                    progress_listener.on_progress(0, len(df_vlm), len(self.best), 6, "Поиск товаров")
                
                try:
                    df_vlm['raw_text_clean'] = df_vlm['raw_text'].apply(clean_ocr_text)
                    df_vlm_match = df_vlm.rename(columns={'raw_text_clean': 'ocr_text'})
                    df_vlm_match = find_top5_matches(df_vlm_match, ocr_col='ocr_text')
                    
                    if progress_listener:
                        progress_listener.on_progress(len(df_vlm), len(df_vlm), len(self.best), 6, "Поиск товаров")
                except Exception as e:
                    print(f"[Product Matcher] Ошибка: {e}")
                    df_vlm_match = df_vlm.copy()
                    for i in range(1, 6):
                        df_vlm_match[f'top{i}'] = None
                        df_vlm_match[f'top{i}_sku'] = None
                
                # Освобождаем DataFrame VLM + product matching перед загрузкой LLM
                if 'df_vlm' in dir() and df_vlm is not None:
                    del df_vlm
                cleanup_vram()
                
                # --- Этап 7: LLM подбор SKU ---
                if progress_listener:
                    progress_listener.on_progress(0, len(df_vlm_match), len(self.best), 7, "LLM подбор SKU")
                
                try:
                    hf_matcher = HFSKUMatcher(
                        model_name='Qwen/Qwen2.5-7B-Instruct',
                        batch_size=8,
                        max_new_tokens=64,
                        temperature=0.1,
                        log_to_file=False
                    )
                    df_final = hf_matcher.process_dataframe(df_vlm_match, ocr_col='ocr_text', output_col='id_sku')
                    hf_matcher.unload_model()
                    del hf_matcher
                    
                    if progress_listener:
                        progress_listener.on_progress(len(df_vlm_match), len(df_vlm_match), len(self.best), 7, "LLM подбор SKU")
                except Exception as e:
                    print(f"[LLM SKU] Ошибка: {e}")
                    df_final = df_vlm_match.copy()
                    df_final['id_sku'] = 'нет'
                    try:
                        if 'hf_matcher' in dir():
                            del hf_matcher
                    except Exception:
                        pass
                
                cleanup_vram()
                
                rows_crops = self._merge_vlm_results(rows_crops, df_final)
                
                if 'df_vlm_match' in dir():
                    del df_vlm_match
                if 'df_final' in dir():
                    del df_final
                if 'df_vlm' in dir():
                    del df_vlm
                
                cleanup_vram()
        
        all_depts = [p['department'] for seg in self.seg_preds for p in seg]
        mode_dept, mode_count = Counter(all_depts).most_common(1)[0] if all_depts else ("", 0)
        
        elapsed_time = time.time() - start_time
        stats = SessionStats(
            total_frames=fi + 1,
            total_detections=len(rows_crops),
            elapsed_time=round(elapsed_time, 2),
            fps=round((fi + 1) / elapsed_time, 1) if elapsed_time > 0 else 0.0,
            mode_department=mode_dept,
            mode_department_count=mode_count
        )
        
        self.repository.save_session(session_tag, stats, rows_crops, os.path.basename(video_path), rotation)
        
        cleanup_vram(verbose=True)
        
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