from __future__ import annotations

import logging
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="src", pythonpath=True)

from scripts.parser import top1_crops
from src.crop_extraction import VideoObjectDetector
from src.ocr_processing import text_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    config = OmegaConf.load('params.yaml')

    detector = VideoObjectDetector(config.main_extraction)

    detector.run()
    
    top1_crops(config.top_k_crops)
    
    text_extraction(config.ocr)

if __name__ == "__main__":
    main()