import torch
import cv2
import argparse
from pathlib import Path
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

parser = argparse.ArgumentParser(description='Подготовка датасета для YOLO через Grounding DINO')
parser.add_argument('input_folder', type=str, help='Папка с видео или изображениями')
parser.add_argument('--output_folder', type=str, default='dataset', help='Папка для YOLO датасета')
parser.add_argument('--interval', type=int, default=5, help='Интервал кадров для видео')
parser.add_argument('--threshold', type=float, default=0.2, help='Box threshold')
parser.add_argument('--text-threshold', type=float, default=0.15, help='Text threshold')
parser.add_argument('--color-filter', action='store_true', help='Фильтр по цвету')
parser.add_argument('--min-area', type=int, default=500, help='Мин. площадь детекции')
parser.add_argument('--train-split', type=float, default=0.8, help='Доля train (0.8 = 80%%)')
args = parser.parse_args()

input_folder = Path(args.input_folder)
dataset_path = Path(args.output_folder)

# Создаём структуру YOLO датасета
(dataset_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(dataset_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(dataset_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(dataset_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

print(f"Входная папка: {input_folder}")
print(f"Выходная папка: {dataset_path}")

video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

video_files = [f for f in input_folder.iterdir() 
               if f.suffix.lower() in video_extensions and not f.name.startswith('~')]
image_files = [f for f in input_folder.iterdir() 
               if f.suffix.lower() in image_extensions and not f.name.startswith('~')]

print(f"Найдено видео: {len(video_files)}")
print(f"Найдено изображений: {len(image_files)}")

print("\nЗагрузка Grounding DINO...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)
gd_model.eval()
print("Модель загружена.\n")

def filter_price_tag_colors(frame, xyxy, min_area=500):
    """Фильтр по цвету: белый/красный/желтый"""
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
        
        lower_red1 = np.array([0, 80, 100])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([150, 80, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(roi, lower_red1, upper_red1), 
                                   cv2.inRange(roi, lower_red2, upper_red2))
        
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([20, 25, 255])
        mask_white = cv2.inRange(roi, lower_white, upper_white)
        
        lower_yellow = np.array([20, 80, 150])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(roi, lower_yellow, upper_yellow)
        
        lower_orange = np.array([10, 80, 150])
        upper_orange = np.array([20, 255, 255])
        mask_yellow = cv2.bitwise_or(mask_yellow, cv2.inRange(roi, lower_orange, upper_orange))
        
        red_ratio = np.sum(mask_red > 0) / roi_size
        white_ratio = np.sum(mask_white > 0) / roi_size
        yellow_ratio = np.sum(mask_yellow > 0) / roi_size
        
        if (red_ratio > 0.03 and white_ratio > 0.1) or \
           (yellow_ratio > 0.05 and white_ratio > 0.1) or \
           (yellow_ratio > 0.05 and red_ratio > 0.03) or \
           red_ratio > 0.15 or white_ratio > 0.25 or yellow_ratio > 0.15:
            filtered_indices.append(i)
    
    return filtered_indices

def detect_price_tags(frame):
    """Grounding DINO детекция"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    text_prompt = "price tag . price label . ценник . этикетка с ценой"
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = gd_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
        target_sizes=[pil_image.size[::-1]]
    )
    
    xyxy = results[0]["boxes"].cpu().numpy()
    confidence = results[0]["scores"].cpu().numpy()
    
    if args.color_filter and len(xyxy) > 0:
        filtered_idx = filter_price_tag_colors(frame, xyxy, args.min_area)
        xyxy = xyxy[filtered_idx]
        confidence = confidence[filtered_idx]
    
    boxes = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w * h >= args.min_area:
            boxes.append([x1, y1, w, h])
    
    return boxes

def boxes_to_yolo(boxes, img_width, img_height):
    """Конвертация в формат YOLO"""
    yolo_boxes = []
    for box in boxes:
        x, y, w, h = box
        x_center = ((x + w / 2) / img_width)
        y_center = ((y + h / 2) / img_height)
        w_norm = w / img_width
        h_norm = h / img_height
        
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        w_norm = max(0, min(1, w_norm))
        h_norm = max(0, min(1, h_norm))
        
        yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    
    return yolo_boxes

all_images = []

def process_image(image_path, frame_num=0):
    """Обработка изображения"""
    frame = cv2.imread(str(image_path))
    if frame is None:
        return False
    
    height, width = frame.shape[:2]
    boxes = detect_price_tags(frame)
    
    img_name = f"img_{frame_num:06d}.jpg"
    img_path = dataset_path / 'images' / 'train' / img_name
    cv2.imwrite(str(img_path), frame)
    
    label_path = dataset_path / 'labels' / 'train' / img_name.replace('.jpg', '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(boxes_to_yolo(boxes, width, height)))
    
    all_images.append(img_name)
    return True

def process_video(video_path):
    """Обработка видео"""
    print(f"Обработка видео: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    processed_count = 0
    total_detections = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % args.interval != 0:
            continue
        
        processed_count += 1
        boxes = detect_price_tags(frame)
        total_detections += len(boxes)
        
        img_name = f"{video_path.stem}_f{frame_count:06d}.jpg"
        img_path = dataset_path / 'images' / 'train' / img_name
        cv2.imwrite(str(img_path), frame)
        
        label_path = dataset_path / 'labels' / 'train' / img_name.replace('.jpg', '.txt')
        with open(label_path, 'w') as f:
            f.write('\n'.join(boxes_to_yolo(boxes, width, height)))
        
        all_images.append(img_name)
        
        if processed_count % 20 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Прогресс: {progress:.1f}% ({frame_count}/{total_frames}), детекций: {total_detections}")
    
    cap.release()
    print(f"  Обработано кадров: {processed_count}, детекций: {total_detections}")
    return total_detections

# Обработка изображений
print("\n" + "="*50)
print("ОБРАБОТКА ИЗОБРАЖЕНИЙ")
print("="*50)

img_count = 0
for img_path in image_files:
    if process_image(img_path, img_count):
        img_count += 1
        print(f"  {img_path.name}: OK")

print(f"Всего изображений: {img_count}")

# Обработка видео
print("\n" + "="*50)
print("ОБРАБОТКА ВИДЕО")
print("="*50)

total_video_detections = 0
for video_path in video_files:
    total_video_detections += process_video(video_path)

# Разделение на train/val
print("\n" + "="*50)
print("РАЗДЕЛЕНИЕ НА TRAIN/VAL")
print("="*50)

if len(all_images) > 0:
    train_images, val_images = train_test_split(
        all_images, 
        train_size=args.train_split, 
        random_state=42
    )
    
    # Перемещаем валидационные изображения
    for img_name in val_images:
        src_img = dataset_path / 'images' / 'train' / img_name
        src_label = dataset_path / 'labels' / 'train' / img_name.replace('.jpg', '.txt')
        
        dst_img = dataset_path / 'images' / 'val' / img_name
        dst_label = dataset_path / 'labels' / 'val' / img_name.replace('.jpg', '.txt')
        
        if src_img.exists():
            shutil.move(str(src_img), str(dst_img))
        if src_label.exists():
            shutil.move(str(src_label), str(dst_label))
    
    print(f"Train: {len(train_images)} изображений")
    print(f"Val: {len(val_images)} изображений")

# Создание data.yaml
print("\n" + "="*50)
print("СОЗДАНИЕ data.yaml")
print("="*50)

data_yaml = f"""train: {(dataset_path / 'images' / 'train').resolve()}
val: {(dataset_path / 'images' / 'val').resolve()}

nc: 1
names:
  - price_tag
"""

with open(dataset_path / 'data.yaml', 'w') as f:
    f.write(data_yaml)

print(f"data.yaml создан: {dataset_path / 'data.yaml'}")
print(data_yaml)

# Статистика
total_images = len(train_images) + len(val_images) if len(all_images) > 0 else img_count
print("\n" + "="*50)
print("ГОТОВО!")
print("="*50)
print(f"Всего изображений: {total_images}")
print(f"Датасет: {dataset_path.resolve()}")
print(f"\nДля обучения:")
print(f"  python train_yolo.py {dataset_path}")
