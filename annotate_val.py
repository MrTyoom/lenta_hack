import json
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Пути
VAL_IMAGES_DIR = Path(r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\dataset_price_tags\images\val")
VAL_LABELS_DIR = Path(r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\dataset_price_tags\labels\val")
MODEL_PATH = r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\runs\detect\yolo_training\price_tags_v1\weights\epoch60.pt"
OUTPUT_DIR = Path(r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\annotated_output")

# Создаем выходные директории
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True)
(OUTPUT_DIR / "images_with_boxes").mkdir(exist_ok=True)

# Загружаем лучшую модель (эпоха 60)
print("Загрузка модели epoch60.pt...")
model = YOLO(MODEL_PATH)

# Получаем имена классов
class_names = model.names

all_annotations = []

# Обрабатываем каждое валидационное изображение
val_images = list(VAL_IMAGES_DIR.glob("*.jpg")) + list(VAL_IMAGES_DIR.glob("*.png"))

print(f"Найдено {len(val_images)} валидационных изображений")

for img_path in val_images:
    img_name = img_path.name
    print(f"Обработка: {img_name}")
    
    # Копируем изображение (если еще не скопировано)
    output_img_path = OUTPUT_DIR / "images" / img_name
    if not output_img_path.exists():
        shutil.copy2(str(img_path), str(output_img_path))
    
    # Запускаем инференс
    results = model.predict(str(img_path), conf=0.25, iou=0.45)
    result = results[0]
    
    boxes = result.boxes
    image_annotations = {
        "image_name": img_name,
        "image_path": str(output_img_path),
        "detections": []
    }
    
    # Создаем текстовый файл с метками
    label_file_path = OUTPUT_DIR / "labels" / img_name.replace(".jpg", ".txt").replace(".png", ".txt")
    
    with open(label_file_path, "w") as f:
        if boxes is not None:
            for box in boxes:
                # Получаем координаты bounding box
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                class_name = class_names[cls]
                
                x1, y1, x2, y2 = xyxy
                
                # Записываем в YOLO формат (нормализованные координаты)
                img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                # Добавляем в JSON
                image_annotations["detections"].append({
                    "class_id": int(cls),
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_normalized": {
                        "x_center": float(x_center),
                        "y_center": float(y_center),
                        "width": float(width),
                        "height": float(height)
                    }
                })
    
    all_annotations.append(image_annotations)
    print(f"  Найдено объектов: {len(image_annotations['detections'])}")
    
    # Рисуем bounding boxes на изображении
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Пытаемся загрузить шрифт
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    for det in image_annotations["detections"]:
        # Координаты уже в пикселях оригинального изображения
        x1, y1, x2, y2 = det["bbox_xyxy"]
        conf = det["confidence"]
        class_name = det["class_name"]
        
        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        
        # Добавляем подпись
        label = f"{class_name}: {conf:.2f}"
        draw.text((x1, y1 - 25), label, fill="green", font=font)
    
    # Сохраняем изображение с боксами
    output_img_boxes_path = OUTPUT_DIR / "images_with_boxes" / img_name
    img.save(output_img_boxes_path, quality=95)

# Сохраняем общий JSON файл
output_json_path = OUTPUT_DIR / "annotations.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump({
        "model_used": "epoch60.pt",
        "model_path": MODEL_PATH,
        "best_epoch": 60,
        "metrics": {
            "mAP50-95": 0.70574,
            "precision": 0.94618,
            "recall": 0.70084
        },
        "total_images": len(all_annotations),
        "annotations": all_annotations
    }, f, indent=2, ensure_ascii=False)

print(f"\nГотово!")
print(f"Модель: epoch60.pt (эпоха 60, mAP50-95 = 0.70574)")
print(f"Изображения: {OUTPUT_DIR / 'images'}")
print(f"Метки: {OUTPUT_DIR / 'labels'}")
print(f"JSON: {output_json_path}")
