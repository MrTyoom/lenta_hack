import cv2
import torch
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Настройки
MODEL_PATH = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\runs\detect\yolo_training\price_tags_v1\weights\epoch60.pt"
INPUT_FOLDER = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\photo"
OUTPUT_FOLDER = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\photo_output"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Входная папка: {INPUT_FOLDER}")
print(f"Выходная папка: {OUTPUT_FOLDER}")

# Загрузка YOLO модели
print("\nЗагрузка YOLO модели...")
model = YOLO(MODEL_PATH)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print("Модель загружена.\n")

# Поиск изображений
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
image_files = [f for f in Path(INPUT_FOLDER).iterdir() 
               if f.suffix.lower() in image_extensions and not f.name.startswith('~')]

print(f"Найдено изображений: {len(image_files)}")

# DataFrame для результатов
data = []

# Обработка изображений
start_time = time.time()
processed = 0

for img_path in image_files:
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"Не удалось прочитать: {img_path.name}")
        continue
    
    height, width = frame.shape[:2]
    
    # Детекция YOLO
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    result = results[0]
    
    # Обработка детекций
    boxes = result.boxes
    if boxes is not None:
        for box_num, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # Добавляем в DataFrame
            data.append({
                'image': str(img_path.name),
                'box_number': box_num,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'w': w,
                'h': h,
                'area': area,
                'confidence': conf,
                'class': cls
            })
            
            # Рисуем прямоугольник
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Подпись
            label = f"price_tag {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Сохраняем изображение с боксами
    cv2.imwrite(f"{OUTPUT_FOLDER}/{img_path.name}", frame)
    
    processed += 1
    if processed % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Обработано: {processed}/{len(image_files)} ({elapsed:.1f} сек)")

# Сохраняем DataFrame
df = pd.DataFrame(data)
df.to_csv(f"{OUTPUT_FOLDER}/detections.csv", index=False, encoding='utf-8-sig')

total_elapsed = time.time() - start_time

print(f"\nГотово! Обработано {processed} изображений")
print(f"Всего найдено ценников: {len(data)}")
print(f"Результаты в папке: {OUTPUT_FOLDER}")
print(f"DataFrame сохранён: {OUTPUT_FOLDER}/detections.csv")
print(f"Общее время обработки: {total_elapsed:.2f} сек ({total_elapsed/60:.2f} мин)")
