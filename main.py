import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Настройки
INPUT_FOLDER = r"photo"
OUTPUT_FOLDER = r"color_filtered_output"
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10
MIN_AREA = 500
COLOR_FILTER = True
SAMPLE_EVERY = 10  # Каждую 10-ю фотографию с подписями

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Входная папка: {INPUT_FOLDER}")
print(f"Выходная папка: {OUTPUT_FOLDER}")

# Загрузка Grounding DINO
print("\nЗагрузка Grounding DINO...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
gd_model.eval()
print("Модель загружена.\n")


def filter_price_tag_colors(frame, xyxy, min_area=500):
    """Фильтр по цвету: белый/красный/желтый с учетом многоцветных ценников"""
    filtered_indices = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue
        
        roi = hsv[y1:y2, x1:x2]
        roi_size = roi.shape[0] * roi.shape[1]
        
        # Красный
        lower_red1 = np.array([0, 50, 80])
        upper_red1 = np.array([14, 255, 255])
        lower_red2 = np.array([145, 50, 80])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(roi, lower_red1, upper_red1), 
                                   cv2.inRange(roi, lower_red2, upper_red2))
        
        # Белый
        lower_white = np.array([0, 0, 150])
        upper_white = np.array([25, 40, 255])
        mask_white = cv2.inRange(roi, lower_white, upper_white)
        
        # Жёлтый + оранжевый
        lower_yellow = np.array([18, 60, 120])
        upper_yellow = np.array([38, 255, 255])
        mask_yellow = cv2.inRange(roi, lower_yellow, upper_yellow)
        lower_orange = np.array([10, 60, 120])
        upper_orange = np.array([18, 255, 255])
        mask_yellow = cv2.bitwise_or(mask_yellow, cv2.inRange(roi, lower_orange, upper_orange))
        
        red_ratio = np.sum(mask_red > 0) / roi_size
        white_ratio = np.sum(mask_white > 0) / roi_size
        yellow_ratio = np.sum(mask_yellow > 0) / roi_size
        
        # Проверяем наличие хотя бы 2-х цветов ИЛИ одного доминирующего
        has_multi_color = (red_ratio > 0.02 and white_ratio > 0.05) or \
                         (yellow_ratio > 0.03 and white_ratio > 0.05) or \
                         (yellow_ratio > 0.03 and red_ratio > 0.02)
        has_single_color = red_ratio > 0.10 or white_ratio > 0.15 or yellow_ratio > 0.10
        
        if has_multi_color or has_single_color:
            filtered_indices.append(i)
    
    return filtered_indices


def detect_price_tags(frame):
    """Grounding DINO детекция ценников"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    text_prompt = "price tag . price label . ценник . этикетка с ценой . красный ценник . желтый ценник"
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = gd_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil_image.size[::-1]]
    )
    
    xyxy = results[0]["boxes"].cpu().numpy()
    confidence = results[0]["scores"].cpu().numpy()
    
    if COLOR_FILTER and len(xyxy) > 0:
        filtered_idx = filter_price_tag_colors(frame, xyxy, MIN_AREA)
        xyxy = xyxy[filtered_idx]
        confidence = confidence[filtered_idx]
    
    boxes = []
    confidences = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w * h >= MIN_AREA:
            boxes.append((x1, y1, w, h))
            confidences.append(float(confidence[i]))
    
    return boxes, confidences


# Обработка изображений
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
image_files = [f for f in Path(INPUT_FOLDER).iterdir() 
               if f.suffix.lower() in image_extensions and not f.name.startswith('~')]

print(f"Найдено изображений: {len(image_files)}")

# DataFrame для результатов
data = []

processed = 0
for idx, img_path in enumerate(image_files):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Не удалось прочитать: {img_path.name}")
        continue
    
    boxes, confidences = detect_price_tags(frame)
    
    # Добавляем в DataFrame
    for box_num, (box, conf) in enumerate(zip(boxes, confidences)):
        x, y, w, h = box
        area = w * h
        data.append({
            'image': str(img_path),
            'box_number': box_num,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'confidence': conf
        })
    
    # Рисуем боксы
    for box_num, (box, conf) in enumerate(zip(boxes, confidences)):
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Рисуем прямоугольник
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Подписываем confidence на каждой SAMPLE_EVERY фотографии
        if (idx + 1) % SAMPLE_EVERY == 0:
            label = f"{conf:.2f}"
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Сохраняем
    cv2.imwrite(f"{OUTPUT_FOLDER}/{img_path.name}", frame)
    
    processed += 1
    if processed % 10 == 0:
        print(f"Обработано: {processed}/{len(image_files)}")

# Сохраняем DataFrame
df = pd.DataFrame(data)
df.to_csv(f"{OUTPUT_FOLDER}/detections.csv", index=False, encoding='utf-8-sig')

print(f"\nГотово! Обработано {processed} изображений")
print(f"Всего найдено ценников: {len(data)}")
print(f"Результаты в папке: {OUTPUT_FOLDER}")
print(f"DataFrame сохранён: {OUTPUT_FOLDER}/detections.csv")
