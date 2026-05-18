import os
import re
import gc
import time
import torch
import shutil
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

CURRENT_CONFIG = "4-bit NF4"

MODEL_PATH = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\VLM MODULE\AVITO"
CROPS_DIR = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\data\tmp\obj_det\best_crops"
OUTPUT_DIR = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\VLM MODULE\vlm_select"

PROMPT = """You are analyzing a Russian supermarket price tag. Study the layout carefully.

ELEMENTS AND THEIR POSITIONS:
- Product name (top-left area, WHITE BACKGROUND): full product name, brand, variety. Located at the top on white background, near the QR-code. Can span MULTIPLE LINES.
- QR-code (top-right corner): square QR code
- Discount size (left side, circle): percentage discount like "-26%" or "-32%"
- Price without card (right side, under QR-code): regular price, smaller font
- Price with card (right side, large font): main price, biggest number on tag
- Barcode (bottom area): horizontal barcode with digits below it
- Article/SKU (under price with card): small number, format digits_digits (e.g. 12345_678)
- Print date/time (very bottom): date and time when tag was printed

Extract ALL visible fields. Return ONLY the values in this exact format with <> as separator:
product_name<>price_without_card<>price_with_card<>promo_price<>barcode<>discount_size<>article<>layout_code<>print_date

Rules:
- promo_price: if there is a separate promo/sale price, otherwise leave empty
- layout_code: character � inside a circle (if present), otherwise leave empty
- article: look for pattern  digits_digits (e.g. 12345_678)
- print_date: full date and time from bottom of tag
- If any field is not visible, leave it empty (do NOT write "not found")

Respond in Russian language."""

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def collect_images_by_folder(crops_dir):
    crops_path = Path(crops_dir)
    folders = {}
    for ext in IMAGE_EXTENSIONS:
        for img_path in crops_path.rglob(f'*{ext}'):
            relative = img_path.relative_to(crops_path)
            folder_name = relative.parts[0]
            if folder_name not in folders:
                folders[folder_name] = []
            folders[folder_name].append({'crop_path': str(img_path), 'crop_name': img_path.name})
    total = sum(len(v) for v in folders.values())
    print(f"Found {total} images in {len(folders)} folders:")
    for fname, imgs in folders.items():
        print(f"  {fname}: {len(imgs)} images")
    return folders

def parse_response(raw_text):
    fields = {'product_name': '', 'price_without_card': '', 'price_with_card': '', 'promo_price': '', 'barcode': '', 'discount_size': '', 'article': '', 'layout_code': '', 'print_date': ''}
    cleaned = re.sub(r'<\|im_start\|>|<\|im_end\|>|<\|vision_start\|>|<\|vision_end\|>', '', raw_text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    cleaned = cleaned.strip()
    parts = cleaned.split('<>')
    field_names = list(fields.keys())
    for i, part in enumerate(parts):
        if i < len(field_names):
            value = part.strip()
            if ':' in value:
                value = value.split(':', 1)[1].strip()
            if value and value.lower() not in ['not found', 'none', 'n/a', '-']:
                fields[field_names[i]] = value
    if not fields['article']:
        m = re.search(r'(\d+_\d+)', cleaned)
        if m: fields['article'] = m.group(1)
    if not fields['barcode']:
        m = re.search(r'\b(\d{8,14})\b', cleaned)
        if m: fields['barcode'] = m.group(1)
    if not fields['discount_size']:
        m = re.search(r'(-?\d+[%\s]*руб|рублей|\d+[%])', cleaned, re.IGNORECASE)
        if m: fields['discount_size'] = m.group(1)
    return fields


def get_configs():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    return {
        "4-bit NF4": {"model_kwargs": {"quantization_config": q, "device_map": "auto"}, "generate_kwargs": {"max_new_tokens": 512}},
        "Fast-64": {"model_kwargs": {"dtype": torch.float16, "device_map": "auto"}, "generate_kwargs": {"max_new_tokens": 64}},
        "Fast-128": {"model_kwargs": {"dtype": torch.float16, "device_map": "auto"}, "generate_kwargs": {"max_new_tokens": 128}},
    }


def load_vlm_model(model_path, config_name="4-bit NF4"):
    configs = get_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    cp = configs[config_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, **cp["model_kwargs"])
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return model, processor


def vlm_predict_crops(crops_df, model, processor, config_name="4-bit NF4"):
    configs = get_configs()
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")
    cp = configs[config_name]
    results_rows = []
    for idx, row in crops_df.iterrows():
        crop_array = row['crop_array']
        img = Image.fromarray(cv2.cvtColor(crop_array, cv2.COLOR_BGR2RGB))
        messages = [{"role": "user", "content": [{"type": "image", "image": img, "min_pixels": 4*28*28, "max_pixels": 1024*28*28}, {"type": "text", "text": PROMPT}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, **cp["generate_kwargs"])
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = parse_response(raw)
        result = {k: v for k, v in row.items() if k != 'crop_array'}
        result.update(parsed)
        results_rows.append(result)
        if (idx + 1) % 5 == 0:
            print(f"  OCR [{idx+1}/{len(crops_df)}]")
    return pd.DataFrame(results_rows)


def run_benchmark():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    folders = collect_images_by_folder(CROPS_DIR)
    if not folders:
        print("No images found. Exiting.")
        return
    config_params = get_configs()[CURRENT_CONFIG]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    load_start = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, local_files_only=True, **config_params["model_kwargs"])
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    all_results = []
    for folder_name, images in folders.items():
        print(f"\nProcessing folder: {folder_name} ({len(images)} images)")
        folder_output_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_output_dir, exist_ok=True)
        folder_results = []
        for idx, img_info in enumerate(images):
            crop_path = img_info['crop_path']
            crop_name = img_info['crop_name']
            print(f"  [{idx+1}/{len(images)}]: {crop_name}")
            try:
                img = Image.open(crop_path).convert('RGB')
                messages = [{"role": "user", "content": [{"type": "image", "image": img, "min_pixels": 4*28*28, "max_pixels": 1024*28*28}, {"type": "text", "text": PROMPT}]}]
                chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
                inf_start = time.time()
                generated_ids = model.generate(**inputs, **config_params["generate_kwargs"])
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                raw_response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                inf_time = time.time() - inf_start
                parsed = parse_response(raw_response)
                result_row = {'config': CURRENT_CONFIG, 'crop_path': crop_path, 'crop_name': crop_name, 'load_time_sec': round(load_time, 2), 'inference_time_sec': round(inf_time, 2), 'product_name': parsed['product_name'], 'price_without_card': parsed['price_without_card'], 'price_with_card': parsed['price_with_card'], 'promo_price': parsed['promo_price'], 'barcode': parsed['barcode'], 'discount_size': parsed['discount_size'], 'article': parsed['article'], 'layout_code': parsed['layout_code'], 'print_date': parsed['print_date'], 'raw_text': raw_response.replace('\n', ' <> ').strip()}
                folder_results.append(result_row)
                all_results.append(result_row)
                shutil.copy2(crop_path, os.path.join(folder_output_dir, crop_name))
                print(f"    Done in {inf_time:.2f}s")
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                error_row = {'config': CURRENT_CONFIG, 'crop_path': crop_path, 'crop_name': crop_name, 'load_time_sec': round(load_time, 2), 'inference_time_sec': -1, 'product_name': '', 'price_without_card': '', 'price_with_card': '', 'promo_price': '', 'barcode': '', 'discount_size': '', 'article': '', 'layout_code': '', 'print_date': '', 'raw_text': f"ERROR: {str(e)}"}
                folder_results.append(error_row)
                all_results.append(error_row)
        folder_df = pd.DataFrame(folder_results)
        folder_df.to_excel(os.path.join(folder_output_dir, f"{folder_name}.xlsx"), index=False)
    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    final_path = os.path.join(OUTPUT_DIR, "results_all.xlsx")
    pd.DataFrame(all_results).to_excel(final_path, index=False)
    print(f"\nDone! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_benchmark()
