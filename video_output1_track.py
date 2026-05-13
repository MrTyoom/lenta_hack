import cv2
import torch
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

# Настройки
MODEL_PATH = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\runs\detect\yolo_training\price_tags_v1\weights\epoch60.pt"
INPUT_FOLDER = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\video_output1"
OUTPUT_FOLDER = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\video_output1"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DETECT_EVERY = 1  # Детекция каждые N кадров

# Опции поворота
ROTATE = False  # False = без поворота, True = 90° против часовой
ROTATION_CODE = cv2.ROTATE_90_COUNTERCLOCKWISE  # Можно менять на ROTATE_90_CLOCKWISE или ROTATE_180

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Входная папка: {INPUT_FOLDER}")
print(f"Выходная папка: {OUTPUT_FOLDER}")
print(f"Поворот: {'90° против часовой' if ROTATE else 'нет'}")

# Загрузка YOLO модели
print("\nЗагрузка YOLO модели...")
model = YOLO(MODEL_PATH)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print("Модель загружена.\n")

# Поиск видео файлов
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
video_files = [f for f in Path(INPUT_FOLDER).iterdir() 
               if f.suffix.lower() in video_extensions and not f.name.startswith('~')]

print(f"Найдено видео: {len(video_files)}")
for v in video_files:
    print(f"  - {v.name}")

# DataFrame для результатов
all_data = []

# Обработка каждого видео
start_time = time.time()
for video_path in video_files:
    print(f"\nОбработка: {video_path.name}")
    video_start = time.time()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Не удалось открыть видео: {video_path.name}")
        continue
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # После поворота ширина и высота меняются местами
    if ROTATE:
        rotated_width, rotated_height = height, width
    else:
        rotated_width, rotated_height = width, height
    
    # Кодек для сохранения
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path(OUTPUT_FOLDER) / f"{video_path.stem}_annotated.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (rotated_width, rotated_height))
    
    frame_count = 0
    processed_frames = 0
    total_detections = 0
    last_boxes = []  # Кэшируем боксы для промежуточных кадров
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Поворот если нужно
        if ROTATE:
            frame = cv2.rotate(frame, ROTATION_CODE)
        
        # Детекция каждые N кадров
        if frame_count % DETECT_EVERY == 0:
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            result = results[0]
            
            # Сохраняем боксы
            boxes = result.boxes
            last_boxes = []
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    last_boxes.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
                    total_detections += 1
        
        # Рисуем боксы (из кэша)
        for x1, y1, x2, y2, conf, cls in last_boxes:
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # Добавляем в DataFrame
            all_data.append({
                'video': video_path.name,
                'frame': frame_count,
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
        
        out.write(frame)
        processed_frames += 1
        
        if processed_frames % 100 == 0:
            progress = (processed_frames / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - video_start
            print(f"  Прогресс: {progress:.1f}% ({processed_frames}/{total_frames}), детекций: {total_detections}")
    
    cap.release()
    out.release()
    
    video_elapsed = time.time() - video_start
    print(f"  Готово! Обработано кадров: {processed_frames}, детекций: {total_detections}")
    print(f"  Время обработки: {video_elapsed:.2f} сек ({video_elapsed/60:.2f} мин)")
    print(f"  Сохранено: {output_path}")

# Сохраняем DataFrame
df = pd.DataFrame(all_data)
df.to_csv(f"{OUTPUT_FOLDER}/detections.csv", index=False, encoding='utf-8-sig')

total_elapsed = time.time() - start_time
print(f"\nГотово! Все видео обработаны.")
print(f"Результаты в папке: {OUTPUT_FOLDER}")
print(f"DataFrame сохранён: {OUTPUT_FOLDER}/detections.csv")
print(f"Общее время обработки: {total_elapsed:.2f} сек ({total_elapsed/60:.2f} мин)")
