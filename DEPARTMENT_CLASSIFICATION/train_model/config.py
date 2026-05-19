"""
Конфигурация для обучения модели
"""

from pathlib import Path

# Пути
DATA_DIR = Path(__file__).parent.parent / "department"

# Исключаем папки которые не нужны
EXCLUDE_FOLDERS = {'', '__pycache__'}

# Модель
MODEL_NAME = 'efficientnet-b0'  # efficientnet-b0, b3, b4 или resnet-50
NUM_CLASSES = 15  # Количество отделов (15 классов)

# Гиперпараметры
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Быстрый старт
LEARNING_RATE = 0.001
IMG_SIZE = 224  # Размер изображения для модели

# Freeze стратегия
FREEZE_EPOCHS = 3  # Сначала обучаем только голову

# Device
DEVICE = 'cuda'  # 'cuda' или 'cpu'

# Аугментации
AUGMENTATION = {
    'horizontal_flip': 0.5,
    'rotation': 15,
    'brightness': 0.2,
    'contrast': 0.2,
}

# Сохранение
SAVE_DIR = Path('./models')
SAVE_DIR.mkdir(exist_ok=True)
