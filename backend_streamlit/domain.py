from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np

@dataclass(frozen=True)
class PriceTag:
    """Бизнес-сущность ценника (Образ результата согласно ТЗ)."""
    filename: str
    product_name: str
    price_default: float
    price_card: float
    price_discount: str
    barcode: int
    discount_amount: str
    id_sku: str
    print_datetime: str
    code: int
    additional_info: str
    color: str
    special_symbols: str
    frame_timestamp: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    qr_code_barcode: int
    price1_qr: float
    price2_qr: str = "нет"
    price3_qr: str = "нет"
    price4_qr: str = "нет"
    wholesale_level_1_count: str = "нет"
    wholesale_level_1_price: str = "нет"
    wholesale_level_2_count: str = "нет"
    wholesale_level_2_price: str = "нет"
    action_price_qr: str = "нет"
    action_code_qr: str = "нет"
    is_problematic: int = 0
    department: str = ""
    department_prob: float = 0.0
    track_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CropCandidate:
    """Кандидат на лучший кроп для трека"""
    score: float
    crop: np.ndarray
    frame_index: int
    confidence: float
    bbox: List[int]


@dataclass
class SessionStats:
    """Метрики производительности прохода робота."""
    total_frames: int = 0
    total_detections: int = 0
    elapsed_time: float = 0.0
    fps: float = 0.0
    mode_department: str = ""
    mode_department_count: int = 0


@dataclass
class ProcessingSession:
    """Метаданные сессии обработки видео"""
    id: int = None
    video_filename: str = ""
    rotation: str = ""
    tag: str = ""
    created_at: str = ""
    total_frames: int = 0
    total_detections: int = 0
    elapsed_time: float = 0.0
    fps: float = 0.0
    mode_department: str = ""
    mode_department_count: int = 0