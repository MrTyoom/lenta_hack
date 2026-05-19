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

MODEL_PATH = str(Path(__file__).parent / "AVITO")
MODEL_HF_ID = "AvitoTech/a-vision"
CROPS_DIR = str(Path(__file__).parent.parent / "data" / "tmp" / "obj_det" / "best_crops")
OUTPUT_DIR = str(Path(__file__).parent / "vlm_select")

PROMPT = """You are a Russian supermarket price tag OCR system. Extract data ONLY.

<CRITICAL_RULES>
- Output ONLY the 9 values separated by <>
- NO explanations, NO comments, NO introductory text
- If you cannot find a field, leave it EMPTY (not "none", not "N/A", not "-")
- Do NOT output any text before or after the data line
- Read Russian text carefully: distinguish И/Й, Ш/Щ, Б/В, П/Л, Ц/Щ, З/Э, О/Ф, Ё/Е
</CRITICAL_RULES>

<OUTPUT_FORMAT>
product_name<>price_without_card<>price_with_card<>promo_price<>barcode<>discount_size<>article<>layout_code<>print_date
</OUTPUT_FORMAT>

<FIELD_DEFINITIONS>
1. product_name: Full product name with brand and variety. Usually on yellow/white background, large font. Include weight/volume if visible (e.g. "Молоко Простоквашино 3.2% 900мл"). Fix obvious OCR errors in Cyrillic (e.g. "MO/\OKO" -> "МОЛОКО").

2. price_without_card: Regular price in RUB.KOP format (e.g. "18999" = 189.99). Usually smaller text near "Цена без карты" or under QR code. If card price is the only price shown, leave this EMPTY.

3. price_with_card: Main price with loyalty card in RUB.KOP (e.g. "14999" = 149.99). Usually the LARGEST number on the tag, often colored red/yellow. This is the price customers with Lenta card pay. Remove currency symbols (₽, руб).

4. promo_price: Special promo/sale price if explicitly marked with a different label (e.g. "Акция", "Суперцена", "Красная цена"). Otherwise EMPTY.

5. barcode: 8-14 digit number from barcode block. Usually starts with 46 (Russia). Strip all spaces, copy exactly as digits.

6. discount_size: Percentage discount with % sign (e.g. "-26%" or "-32%"). Often in colored circle/badge. Include minus sign if present. EMPTY if no discount shown.

7. article: Internal SKU code, format DIGITS_DIGITS (e.g. "12345_678"). Usually printed in small font near barcode or at the bottom right. Sometimes labeled "Арт." or "Код". If only one number visible, use "DIGITS_" or "_DIGITS".

8. layout_code: Single character in a circle (usually at the top or bottom). Б, В, Г, Д, etc. If not present, EMPTY.

9. print_date: Date and time from the bottom line of the tag. Format: YYYY-MM-DD HH:MM:SS. Normalize: convert "24.12.01" -> "2024-12-01", "01.12.24" -> "2024-12-01". EMPTY if no date visible.
</FIELD_DEFINITIONS>

<PRICE_FORMAT_EXAMPLES>
- "18999" = 189.99 RUB -> output "18999" (no decimal point, no currency)
- "75" = 75.00 RUB -> output "7500" (pad to kopecks)
- "149,99" on tag -> output "14999"
- "149.99" on tag -> output "14999"
</PRICE_FORMAT_EXAMPLES>

<EXAMPLES>
Example 1 (full tag with all fields):
Водка Белое Березка Премиум<>18999<>14999<>12999<>4600000123456<>-26%<>12345_678<>Б<>2024-12-01 14:30:00

Example 2 (no promo price, no layout code):
Молоко Домик в Деревне 3.2% 900мл<>8500<>7500<><>4600123456789<>-15%<>98765_432<><>2024-12-01 10:15:00

Example 3 (missing fields - leave empty):
Хлеб Бородинский нарезка<>4500<>3800<><><><><><><>2024-12-01 08:00:00

Example 4 (card price is the only price shown):
Сыр Российский 200г<><>19999<><>4607000001234<><>55555_333<><>2024-12-01 09:30:00

Example 5 (promo price present, no regular price):
Колбаса Докторская Велком<><>29999<>24999<>4600000888777<>-17%<>11111_222<>Д<>2024-11-30 16:45:00
</EXAMPLES>

<NEGATIVE_EXAMPLES>
WRONG: "На изображении представлен ценник..."
WRONG: "Внимательно изучив изображение..."
WRONG: "Продукция_название<>..." (underscore literals!)
WRONG: Any text before or after the data line
WRONG: "Молоко 3.2%<>85<>75<>-15%<><><><><>" (price values without kopecks!)
WRONG: "none<>N/A<>N/A<><><><><><><>" (use EMPTY instead of none/N/A!)
</NEGATIVE_EXAMPLES>

Now extract data from this image. Output ONLY the 9 values:"""

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

ARTICLE_STUBS = {'12345_678', '98765_432', '55555_333', '11111_222', '421'}

BARCODE_STUBS = {'4600000123456', '4600000888777', '4607000001234', '4600123456789', '4210000000000'}


def _is_field_name(val):
    if not val:
        return True
    return val.strip().lower() in {f.lower() for f in FIELD_NAMES_SET}


def _is_article_stub(val):
    if not val:
        return True
    return val.strip() in ARTICLE_STUBS


def _is_barcode_stub(val):
    if not val:
        return True
    return val.strip() in BARCODE_STUBS


def parse_response(raw_text):
    fields = {'product_name': '', 'price_without_card': '', 'price_with_card': '', 'promo_price': '', 'barcode': '', 'discount_size': '', 'article': '', 'layout_code': '', 'print_date': ''}
    
    cleaned = re.sub(r'<\|im_start\|>|<\|im_end\|>|<\|vision_start\|>|<\|vision_end\|>', '', raw_text)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
    
    noise_patterns = [
        r'На\s+изображении\s+представлен[ао]?\s*\.?',
        r'Внимательно\s+изучив\s+изображение',
        r'Вот\s+данные',
        r'Эти\s+данные\s+соответствуют',
        r'извлеченные\s+из\s+описания',
        r'согласно\s+заданным\s+правилам',
        r'как\s+было\s+запрошено',
        r'представлена\s+информация',
        r'В\s+ответе\s+использованы',
    ]
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    lines = cleaned.split('\n')
    data_lines = [line for line in lines if '<>' in line]
    if not data_lines:
        return fields
    
    data_line = data_lines[0].strip()
    
    parts = data_line.split('<>')
    field_names = list(fields.keys())
    
    header_detected = _is_field_name(parts[0].strip()) if parts else False
    
    for i, part in enumerate(parts):
        if i >= len(field_names):
            break
        value = part.strip()
        
        if ':' in value and i > 0:
            value = value.split(':', 1)[1].strip()
        
        if value and value.lower() in ['not found', 'none', 'n/a', '-']:
            value = ''
        
        if _is_field_name(value):
            value = ''
        
        if i == 0 and len(value) > 100:
            value = value[:100]
        
        fields[field_names[i]] = value
    
    if header_detected and not fields['product_name']:
        for line in data_lines[1:]:
            extra = line.strip()
            if '<>' in extra:
                extra_parts = extra.split('<>')
                for p in extra_parts:
                    p = p.strip()
                    if p and not _is_field_name(p) and not re.match(r'^[\d.,%-]+$', p):
                        fields['product_name'] = p
                        break
            elif extra and not _is_field_name(extra) and not re.match(r'^[\d.,%-]+$', extra):
                fields['product_name'] = extra
                break
    
    if not fields['article'] or _is_article_stub(fields['article']):
        fields['article'] = ''
    
    if not fields['barcode'] or _is_barcode_stub(fields['barcode']):
        m = re.search(r'\b(\d{8,14})\b', data_line)
        if m and not _is_barcode_stub(m.group(1)):
            fields['barcode'] = m.group(1)
        elif _is_barcode_stub(fields['barcode']):
            fields['barcode'] = ''
    
    if fields['discount_size']:
        m = re.search(r'(-?\d+)%', fields['discount_size'])
        if m:
            fields['discount_size'] = f"{m.group(1)}%"
    
    for key in fields:
        fields[key] = fields[key].replace('<0x0A>', ' ').replace('▁', ' ').strip()
    
    return fields


def get_configs():
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, llm_int8_enable_fp32_cpu_offload=True)
    q8 = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    return {
        "4-bit NF4": {"model_kwargs": {"quantization_config": q, "device_map": "sequential"}, "generate_kwargs": {"max_new_tokens": 512}},
        "High Quality": {"model_kwargs": {"quantization_config": q, "device_map": "sequential"}, "generate_kwargs": {"max_new_tokens": 512, "do_sample": False, "temperature": 0.1}, "vision_kwargs": {"min_pixels": 4*28*28, "max_pixels": 2048*28*28}},
        "High Quality 8-bit": {"model_kwargs": {"quantization_config": q8, "device_map": "sequential"}, "generate_kwargs": {"max_new_tokens": 512, "do_sample": False, "temperature": 0.1}, "vision_kwargs": {"min_pixels": 4*28*28, "max_pixels": 2048*28*28}},
        "High Quality 8-bit Ultra": {"model_kwargs": {"quantization_config": q8, "device_map": "sequential"}, "generate_kwargs": {"max_new_tokens": 384, "do_sample": False, "temperature": 0.1, "repetition_penalty": 1.2}, "vision_kwargs": {"min_pixels": 8*28*28, "max_pixels": 3072*28*28}},
        "High Quality 8-bit Pro": {"model_kwargs": {"quantization_config": q8, "device_map": "sequential"}, "generate_kwargs": {"max_new_tokens": 384, "do_sample": False, "temperature": 0.05, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}, "vision_kwargs": {"min_pixels": 12*28*28, "max_pixels": 4096*28*28}},
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
    if os.path.isdir(model_path):
        model = AutoModelForImageTextToText.from_pretrained(model_path, local_files_only=True, **cp["model_kwargs"])
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_HF_ID, **cp["model_kwargs"])
        processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
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
        vk = cp.get("vision_kwargs", {"min_pixels": 4*28*28, "max_pixels": 1024*28*28})
        messages = [{"role": "user", "content": [{"type": "image", "image": img, **vk}, {"type": "text", "text": PROMPT}]}]
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[chat_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
        generated_ids = model.generate(**inputs, **cp["generate_kwargs"])
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        parsed = parse_response(raw)
        result = {k: v for k, v in row.items() if k != 'crop_array'}
        result.update(parsed)
        result['raw_text'] = raw
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
    if os.path.isdir(MODEL_PATH):
        model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH, local_files_only=True, **config_params["model_kwargs"])
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(MODEL_HF_ID, **config_params["model_kwargs"])
        processor = AutoProcessor.from_pretrained(MODEL_HF_ID)
        model.save_pretrained(MODEL_PATH)
        processor.save_pretrained(MODEL_PATH)
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
