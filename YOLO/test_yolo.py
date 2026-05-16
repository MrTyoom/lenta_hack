import torch
from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import numpy as np

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

parser = argparse.ArgumentParser(description='Тестирование обученной YOLO модели')
parser.add_argument('model_path', type=str, help='Путь к модели (.pt файл)')
parser.add_argument('--source', type=str, default='0', help='Источник: 0=веб-камера, путь к фото/видео')
parser.add_argument('--conf', type=float, default=0.25, help='Порог уверенности')
parser.add_argument('--iou', type=float, default=0.45, help='Порог NMS IoU')
parser.add_argument('--save', action='store_true', help='Сохранить результат')
parser.add_argument('--output', type=str, default='output', help='Папка для сохранения')
args = parser.parse_args()

print(f"\n🔍 Загрузка модели: {args.model_path}")
model = YOLO(args.model_path)
print(f"✅ Модель загружена")

print(f"\n📹 Источник: {args.source}")

if args.source.isdigit():
    source = int(args.source)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {source}")
        exit(1)
    
    print(f"✅ Камера открыта")
    print(f"   Нажмите 'q' для выхода\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
        
        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLO Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
else:
    source_path = Path(args.source)
    
    if not source_path.exists():
        print(f"❌ Файл не найден: {source_path}")
        exit(1)
    
    if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        print(f"📷 Обработка изображения: {source_path.name}")
        
        frame = cv2.imread(str(source_path))
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
        
        annotated_frame = results[0].plot()
        
        if args.save:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"result_{source_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"✅ Сохранено: {output_path}")
        
        cv2.imshow('Result', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"🎬 Обработка видео: {source_path.name}")
        
        cap = cv2.VideoCapture(str(source_path))
        
        if not cap.isOpened():
            print(f"❌ Не удалось открыть видео")
            exit(1)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Размер: {width}x{height}, FPS: {fps}, Фреймов: {total_frames}")
        
        output_path = None
        out = None
        
        if args.save:
            output_dir = Path(args.output)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"result_{source_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            results = model(frame, conf=args.conf, iou=args.iou, verbose=False)
            annotated_frame = results[0].plot()
            
            if out:
                out.write(annotated_frame)
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")
        
        cap.release()
        if out:
            out.release()
            print(f"✅ Видео сохранено: {output_path}")
        
        print(f"\n✅ Обработка завершена")
