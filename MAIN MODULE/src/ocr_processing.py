import os
import cv2
import pandas as pd
import glob

from paddleocr import PaddleOCR
from omegaconf import OmegaConf

def vlm_inference():
    pass

def text_extraction(cfg: OmegaConf) -> None:
    ocr = PaddleOCR(
        lang='ru',
        
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,

        text_detection_model_name=None,
        precision='fp16',
        
        # для GPU
        # device='cuda:0',
        # enable_hpi=True,
        # use_tensorrt=True,
        
        # для CPU
        enable_mkldnn=True,
        cpu_threads=cfg.cpu_threads,
        text_recognition_batch_size=cfg.text_recognition_batch_size,
        
        ocr_version='PP-OCRv5'
    )


    image_paths = sorted(os.listdir(cfg.input_folder))

    output_lines = []
    for ind, image_name in enumerate(image_paths):
        image_path = os.path.join(cfg.input_folder, image_name)
        
        crop = cv2.imread(image_path)
        
        try:
            if float(image_path.split('score')[-1]) < cfg.conf_threshold:
                # results = vlm_inference(crop)
                pass
            else:
                results = ocr.predict(crop)
        except ValueError:
            continue
            
        output_lines.append(f'Изображение: {image_path}')

        for _, result in enumerate(results):

            texts = result.get('rec_texts', [])

            output_lines.append(
                f' {" ".join(texts)}'
            )
            
            output_lines.append('=' * 50)
            
    with open('data/ocr_results_on_best_crops.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    