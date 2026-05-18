"""
Скрипт для предсказания отдела по изображению
"""

import torch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json
import sys

from model import get_model


class DepartmentPredictor:
    """Предсказатель отдела по изображению"""
    
    def __init__(self, model_path, class_names_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Загружаем конфиг
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        
        # Загружаем классы
        with open(class_names_path, 'r', encoding='utf-8') as f:
            self.class_names = json.load(f)
        
        # Создаем модель
        self.model = get_model(
            self.config['model_name'],
            num_classes=self.config['num_classes'],
            pretrained=False
        )
        
        # Загружаем веса
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Трансформации
        self.transform = A.Compose([
            A.Resize(self.config['img_size'], self.config['img_size']),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        print(f"Модель загружена: {self.config['model_name']}")
        print(f"Классов: {len(self.class_names)}")
        print(f"Устройство: {self.device}")
    
    def predict(self, image_path, top_k=3):
        """
        Предсказать класс для изображения
        
        Args:
            image_path: Путь к изображению
            top_k: Количество лучших предсказаний
        
        Returns:
            Список кортежей (class_name, probability)
        """
        # Открываем изображение
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Применяем трансформации
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Топ-K предсказаний
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx.item()]
            results.append((class_name, prob.item() * 100))
        
        return results


def main():
    """Предсказание для одного изображения"""
    
    if len(sys.argv) < 2:
        print("Использование: python predict_single.py <путь_к_изображению>")
        print("Пример: python predict_single.py C:\\path\\to\\image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"Ошибка: Файл не найден: {image_path}")
        sys.exit(1)
    
    models_dir = Path('./models')
    
    # Загружаем модель
    predictor = DepartmentPredictor(
        models_dir / 'best_model.pth',
        models_dir / 'class_names.json'
    )
    
    # Делаем предсказание
    print(f"\nАнализ изображения: {image_path}")
    print("-" * 60)
    
    predictions = predictor.predict(image_path)
    
    print(f"\nРезультат:")
    for i, (class_name, prob) in enumerate(predictions, 1):
        if i == 1:
            print(f"  [RESULT] ОТДЕЛ: {class_name} ({prob:.1f}%)")
        else:
            print(f"           {class_name} ({prob:.1f}%)")
    
    print("-" * 60)
    print(f"\nИтоговый отдел: {predictions[0][0]}")


if __name__ == '__main__':
    main()
