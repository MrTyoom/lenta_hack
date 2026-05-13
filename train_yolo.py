import torch
from ultralytics import YOLO
import argparse
from pathlib import Path
import yaml


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser(description='Обучение YOLOv8n на детекцию ценников')
    parser.add_argument('dataset_path', type=str, help='Путь к папке с датасетом (где data.yaml)')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640, help='Размер изображения')
    parser.add_argument('--device', type=int, default=0, help='GPU device (0, 1, 2...)')
    parser.add_argument('--workers', type=int, default=4, help='Количество workers для загрузки данных')
    parser.add_argument('--name', type=str, default='price_tag_yolo8n', help='Название эксперимента')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt', help='Предобученная модель')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    data_yaml_path = dataset_path / 'data.yaml'

    if not data_yaml_path.exists():
        print(f"❌ Файл data.yaml не найден: {data_yaml_path}")
        print("Сначала запустите prepare_dataset.py")
        exit(1)

    print(f"\n✅ Датасет найден: {dataset_path}")
    print(f"   data.yaml: {data_yaml_path}")

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        print(f"\n📊 Конфигурация датасета:")
        print(f"   Train: {data_config.get('train', 'N/A')}")
        print(f"   Val: {data_config.get('val', 'N/A')}")
        print(f"   Классов: {data_config.get('nc', 'N/A')}")
        print(f"   Имена: {data_config.get('names', 'N/A')}")

    print(f"\n🚀 Загрузка модели YOLOv8n...")
    model = YOLO(args.pretrained)
    print(f"✅ Модель загружена: {args.pretrained}")

    print(f"\n📈 Начало обучения...")
    print(f"   Эпох: {args.epochs}")
    print(f"   Batch: {args.batch}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Device: CUDA:{args.device}")

    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        name=args.name,
        project='yolo_training',
        exist_ok=True,
        verbose=True,
        save=True,
        save_period=10,
        plots=True,
        amp=True,
        optimizer='auto',
        patience=50,
        seed=42
    )

    print(f"\n{'='*50}")
    print(f"✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*50}")

    best_model_path = f"yolo_training/{args.name}/weights/best.pt"
    last_model_path = f"yolo_training/{args.name}/weights/last.pt"

    print(f"\n📦 Сохранённые модели:")
    print(f"   Лучшая: {best_model_path}")
    print(f"   Последняя: {last_model_path}")

    print(f"\n📊 Метрики:")
    if hasattr(results, 'results_dict'):
        for key, value in results.results_dict.items():
            print(f"   {key}: {value:.4f}")

    print(f"\n🎯 Для использования:")
    print(f"   from ultralytics import YOLO")
    print(f"   model = YOLO('{best_model_path}')")
    print(f"   results = model('image.jpg')")

    print(f"\n📈 Для тестирования:")
    print(f"   python test_yolo.py {best_model_path}")


if __name__ == '__main__':
    main()
