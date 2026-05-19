import os
import sys
import re
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
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import base64


def _encode_crop_to_base64(crop):
    if crop is None:
        return None
    _, buffer = cv2.imencode('.png', crop)
    return base64.b64encode(buffer).decode('utf-8')


FINAL_COLUMNS = [
    'filename', 'product_name', 'price_default', 'price_card', 'price_discount',
    'barcode', 'discount_amount', 'id_sku', 'print_datetime', 'code',
    'additional_info', 'color', 'special_symbols', 'frame_timestamp',
    'x_min', 'y_min', 'x_max', 'y_max',
    'qr_code_barcode', 'price1_qr', 'price2_qr', 'price3_qr', 'price4_qr',
    'wholesale_level_1_count', 'wholesale_level_1_price',
    'wholesale_level_2_count', 'wholesale_level_2_price',
    'action_price_qr', 'action_code_qr',
]

ARTICLE_STUBS = {'12345_678', '98765_432', '55555_333', '11111_222', '421'}
BARCODE_STUBS = {'4600000123456', '4600000888777', '4607000001234', '4600123456789', '4210000000000'}

FIELD_NAMES_SET = {
    'product_name', 'price_without_card', 'price_with_card', 'promo_price',
    'barcode', 'discount_size', 'article', 'layout_code', 'print_date',
    'product name', 'price without card', 'price with card', 'promo price',
    'discount size', 'layout code', 'print date',
    'продукт_название', 'шифр', 'цена_без_карты', 'цена_с_картой',
    'акционная_цена', 'размер_скидки', 'размер_скидаки', 'артикул',
    'код_расположению', 'код_расположение', 'дата_печати',
    'продукт название', 'цена без карты', 'цена с картой',
    'акционная цена', 'размер скидки', 'размер скидаки',
    'код расположению', 'код расположение', 'дата печати',
}
FIELD_NAMES_LOWER = {f.lower() for f in FIELD_NAMES_SET}


def validate_price(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return None
    s = str(val).replace(',', '.').replace(' ', '').replace('▁', '').strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def validate_barcode(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return None
    digits = re.sub(r'\D', '', str(val))
    if 8 <= len(digits) <= 14 and digits not in BARCODE_STUBS:
        return digits
    return None


def clean_layout_code(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return None
    s = str(val).strip().upper()
    if len(s) == 1 and s.isalpha():
        return s
    if len(s) <= 3:
        for ch in s:
            if ch.isalpha() and ch.upper() in 'АБВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЭЮЯ':
                return ch.upper()
    return None


def clean_print_datetime(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return None
    s = str(val).strip()
    if s.lower().startswith('print_date'):
        s = s[len('print_date'):].strip(': ').strip()
    m = re.search(r'(\d{4}[-.]\d{2}[-.]\d{2})\s*(\d{2}:\d{2}:\d{2})?', s)
    if m:
        date_part = m.group(1).replace('.', '-')
        time_part = m.group(2) or '00:00:00'
        return f"{date_part} {time_part}"
    return s if s else None


def _is_field_name(val):
    if pd.isna(val) or val is None:
        return True
    s = str(val).strip()
    if s == '':
        return True
    return s.lower() in FIELD_NAMES_LOWER


def _clean_str(val):
    if pd.isna(val) or val is None:
        return None
    s = str(val).replace('<0x0A>', ' ').replace('▁', ' ').strip()
    return s if s else None


def _is_article_stub(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return True
    return str(val).strip() in ARTICLE_STUBS


def is_promo_text(val):
    if pd.isna(val) or val is None or str(val).strip() == '':
        return False
    s = str(val).replace('▁', ' ').strip()
    try:
        float(s.replace(',', '.').replace(' ', ''))
        return False
    except (ValueError, TypeError):
        return bool(s)


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

    def _deduplicate_crops(self, df_crops, threshold=0.99):
        if df_crops.empty or 'crop_array' not in df_crops.columns:
            return df_crops
        valid = df_crops[df_crops['crop_array'].notna()]
        if len(valid) <= 1:
            return df_crops
        
        def img_to_vec(arr):
            img = Image.fromarray(arr).convert("RGB").resize((128, 128))
            vec = np.array(img).flatten().astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8)
        
        vectors = np.stack(valid['crop_array'].apply(img_to_vec).values)
        sim_matrix = cosine_similarity(vectors)
        upper = np.triu(sim_matrix, k=1)
        rows, cols = np.where(upper >= threshold)
        to_drop = set(cols.tolist())
        if to_drop:
            df_crops = df_crops.drop(index=valid.index[list(to_drop)]).reset_index(drop=True)
        
        return df_crops

    def _clean_vlm_results(self, df):
        vlm_cols = ['product_name', 'price_without_card', 'price_with_card', 'promo_price',
                     'barcode', 'discount_size', 'article', 'layout_code', 'print_date']
        
        for col in vlm_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if _is_field_name(x) else x)
        
        for col in vlm_cols:
            if col in df.columns:
                df[col] = df[col].apply(_clean_str)
        
        if 'product_name' in df.columns:
            df['product_name'] = df['product_name'].apply(
                lambda x: str(x).replace('_', ' ').strip() if pd.notna(x) and x is not None else None
            )
        
        for col in vlm_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if (isinstance(x, str) and x.strip() == '') else x)
        
        if 'llm_product_name' in df.columns:
            mask = df['llm_product_name'].notna() & (df['llm_product_name'] != '')
            df.loc[mask, 'product_name'] = df.loc[mask, 'llm_product_name']
        
        if 'promo_price' in df.columns:
            if 'special_symbols' not in df.columns:
                df['special_symbols'] = None
            text_mask = df['promo_price'].apply(is_promo_text)
            df.loc[text_mask, 'special_symbols'] = df.loc[text_mask, 'promo_price']
            df.loc[text_mask, 'promo_price'] = None
        
        if 'price_without_card' in df.columns:
            df['price_default'] = df['price_without_card'].apply(validate_price)
        if 'price_with_card' in df.columns:
            df['price_card'] = df['price_with_card'].apply(validate_price)
        if 'promo_price' in df.columns:
            df['price_discount'] = df['promo_price'].apply(validate_price)
        if 'discount_size' in df.columns:
            df['discount_amount'] = df['discount_size'].apply(
                lambda x: x if pd.notna(x) and str(x).strip() else None
            )
        if 'layout_code' in df.columns:
            df['code'] = df['layout_code'].apply(clean_layout_code)
        if 'print_date' in df.columns:
            df['print_datetime'] = df['print_date'].apply(clean_print_datetime)
        
        if 'article' in df.columns:
            no_sku_mask = df['id_sku'].isna()
            valid_article_mask = df['article'].apply(lambda x: not _is_article_stub(x))
            fallback_mask = no_sku_mask & valid_article_mask
            df.loc[fallback_mask, 'id_sku'] = df.loc[fallback_mask, 'article']
        
        df['barcode'] = df['barcode'].apply(validate_barcode)
        
        df_result = pd.DataFrame()
        for col in FINAL_COLUMNS:
            if col in df.columns:
                df_result[col] = df[col].values
            else:
                df_result[col] = None
        
        if 'SYS_trash' in df.columns:
            trash_mask = df['SYS_trash'] == True
            keep_cols = {'filename', 'frame_timestamp', 'x_min', 'y_min', 'x_max', 'y_max'}
            null_cols = [c for c in FINAL_COLUMNS if c not in keep_cols]
            if trash_mask.any():
                for col in null_cols:
                    if col in df_result.columns:
                        df_result.loc[trash_mask, col] = None
        
        return df_result

    def execute(self, video_path: str, rotation: str, session_tag: str, progress_listener: Optional[ProgressListener] = None) -> Tuple[pd.DataFrame, SessionStats]:
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
        
        if self.is_test and len(self.best) >= 1:
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
        
        # Build DataFrame with crop_array for deduplication
        rows_data = []
        crop_map = {}
        for track_id, candidates in self.best.items():
            for rank, c in enumerate(candidates):
                crop_map[(track_id, rank + 1)] = c.crop
        
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
                
                rows_data.append({
                    'filename': os.path.basename(video_path),
                    'SYS_track_id': track_id,
                    'SYS_rank': rank + 1,
                    'SYS_score': round(c.score, 1),
                    'SYS_confidence': round(c.confidence, 3),
                    'SYS_trash': is_trash,
                    'SYS_quality_confidence': round(quality_conf, 1),
                    'x_min': c.bbox[0],
                    'y_min': c.bbox[1],
                    'x_max': c.bbox[2],
                    'y_max': c.bbox[3],
                    'color': crop_color,
                    'department': dept_mode,
                    'department_prob': dept_prob,
                    'frame_timestamp': int(c.frame_index / self.video_fps * 1000),
                    'crop_image': c.crop,
                })
        
        df_crops = pd.DataFrame(rows_data)
        df_crops['crop_array'] = df_crops['crop_image'].apply(lambda x: x if x is not None else None)
        
        # Deduplication
        df_crops = self._deduplicate_crops(df_crops, threshold=0.99)
        
        # --- Этап 5: VLM OCR ---
        non_trash = df_crops[df_crops['SYS_trash'] == False].copy()
        
        if not non_trash.empty:
            if progress_listener:
                progress_listener.on_progress(0, len(non_trash), len(df_crops), 5, "VLM OCR распознавание")
            
            vlm_input_df = non_trash[['SYS_track_id', 'SYS_rank', 'SYS_score', 'SYS_confidence', 'SYS_trash',
                                       'x_min', 'y_min', 'x_max', 'y_max', 'crop_array']].copy()
            
            df_vlm = None
            try:
                vlm_model, vlm_processor = load_vlm_model(self.vlm_model_path, config_name=self.vlm_config_name)
                df_vlm = vlm_predict_crops(vlm_input_df, vlm_model, vlm_processor, config_name=self.vlm_config_name)
                
                if progress_listener:
                    progress_listener.on_progress(len(non_trash), len(non_trash), len(df_crops), 5, "VLM OCR распознавание")
            except Exception as e:
                print(f"[VLM] Ошибка: {e}")
            finally:
                if 'vlm_model' in dir():
                    del vlm_model
                if 'vlm_processor' in dir():
                    del vlm_processor
                cleanup_vram()
            
            # --- Этап 6: Поиск товаров ---
            if df_vlm is not None and not df_vlm.empty:
                if progress_listener:
                    progress_listener.on_progress(0, len(df_vlm), len(df_crops), 6, "Поиск товаров")
                
                df_vlm_match = None
                try:
                    df_vlm['raw_text_clean'] = df_vlm['raw_text'].apply(clean_ocr_text)
                    df_vlm_match = df_vlm.rename(columns={'raw_text_clean': 'ocr_text'})
                    df_vlm_match = find_top5_matches(df_vlm_match, ocr_col='ocr_text')
                    
                    if progress_listener:
                        progress_listener.on_progress(len(df_vlm), len(df_vlm), len(df_crops), 6, "Поиск товаров")
                except Exception as e:
                    print(f"[Product Matcher] Ошибка: {e}")
                    df_vlm_match = df_vlm.copy()
                    for i in range(1, 6):
                        df_vlm_match[f'top{i}'] = None
                        df_vlm_match[f'top{i}_sku'] = None
                
                if 'df_vlm' in dir() and df_vlm is not None:
                    del df_vlm
                cleanup_vram()
                
                # --- Этап 7: LLM подбор SKU ---
                if df_vlm_match is not None and not df_vlm_match.empty:
                    if progress_listener:
                        progress_listener.on_progress(0, len(df_vlm_match), len(df_crops), 7, "LLM подбор SKU")
                    
                    df_final = None
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
                            progress_listener.on_progress(len(df_vlm_match), len(df_vlm_match), len(df_crops), 7, "LLM подбор SKU")
                    except Exception as e:
                        print(f"[LLM SKU] Ошибка: {e}")
                        df_final = df_vlm_match.copy()
                        df_final['id_sku'] = None
                        try:
                            if 'hf_matcher' in dir():
                                del hf_matcher
                        except Exception:
                            pass
                    
                    cleanup_vram()
                    
                    # Merge VLM + LLM results into main DataFrame
                    if df_final is not None and not df_final.empty:
                        # Add llm_product_name from top1 match
                        if 'top1' in df_final.columns:
                            df_final['llm_product_name'] = df_final['top1']
                        
                        # Join df_final columns back to non-trash rows by SYS_track_id
                        merge_cols = ['SYS_track_id'] + [c for c in df_final.columns if c in
                            ['product_name', 'price_without_card', 'price_with_card', 'promo_price',
                             'barcode', 'discount_size', 'article', 'layout_code', 'print_date',
                             'id_sku', 'llm_product_name', 'raw_text']]
                        available_merge = [c for c in merge_cols if c in df_final.columns]
                        vlm_result = df_final[available_merge].copy()
                        
                        non_trash_idx = non_trash.index
                        df_crops = df_crops.merge(
                            vlm_result,
                            on='SYS_track_id',
                            how='left',
                            suffixes=('', '_vlm')
                        )
                    
                    if 'df_vlm_match' in dir():
                        del df_vlm_match
                    if 'df_final' in dir():
                        del df_final
                    cleanup_vram()
        
        # Clean VLM results and build final output
        df_crops = self._clean_vlm_results(df_crops)
        
        all_depts = [p['department'] for seg in self.seg_preds for p in seg]
        mode_dept, mode_count = Counter(all_depts).most_common(1)[0] if all_depts else ("", 0)
        
        elapsed_time = time.time() - start_time
        stats = SessionStats(
            total_frames=fi + 1,
            total_detections=len(df_crops),
            elapsed_time=round(elapsed_time, 2),
            fps=round((fi + 1) / elapsed_time, 1) if elapsed_time > 0 else 0.0,
            mode_department=mode_dept,
            mode_department_count=mode_count
        )
        
        # Extract bad crops for display before dropping crop_image
        self.bad_crops_data = []
        if 'crop_image' in df_crops.columns and 'SYS_trash' in df_crops.columns:
            bad = df_crops[df_crops['SYS_trash'] == True]
            for _, row in bad.iterrows():
                if row.get('crop_image') is not None:
                    self.bad_crops_data.append({
                        'track_id': row.get('SYS_track_id', ''),
                        'confidence': row.get('SYS_quality_confidence', 0),
                        'image_base64': _encode_crop_to_base64(row['crop_image']),
                        'color': row.get('color', ''),
                        'product_name': row.get('product_name', ''),
                    })
        
        # Drop crop_array/crop_image before saving (not serializable)
        if 'crop_image' in df_crops.columns:
            df_crops = df_crops.drop(columns=['crop_image'])
        if 'crop_array' in df_crops.columns:
            df_crops = df_crops.drop(columns=['crop_array'])
        
        # Ensure only FINAL_COLUMNS + SYS columns present for DB saving
        db_cols = FINAL_COLUMNS + ['SYS_trash', 'SYS_quality_confidence', 'department', 'department_prob', 'track_id', 'is_problematic']
        for col in db_cols:
            if col not in df_crops.columns:
                df_crops[col] = None
        
        rows_crops = df_crops.to_dict('records')
        
        self.repository.save_session(session_tag, stats, rows_crops, os.path.basename(video_path), rotation)
        
        cleanup_vram(verbose=True)
        
        return df_crops, stats


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