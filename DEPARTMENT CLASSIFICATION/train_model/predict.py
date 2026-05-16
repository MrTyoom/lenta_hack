"""
Скрипт для предсказания на обученной модели
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import json

from model import get_model


class Predictor:
    """Предсказатель для классификации изображений"""
    
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
        print(f"Классы: {len(self.class_names)}")
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
    
    def predict_batch(self, image_paths, top_k=3):
        """Предсказать для нескольких изображений"""
        results = []
        for img_path in image_paths:
            try:
                pred = self.predict(img_path, top_k)
                results.append({
                    'path': str(img_path),
                    'predictions': pred
                })
            except Exception as e:
                results.append({
                    'path': str(img_path),
                    'error': str(e)
                })
        return results


def main():
    """Тест предсказания"""
    
    models_dir = Path('./models')
    
    # Загружаем модель
    predictor = Predictor(
        models_dir / 'best_model.pth',
        models_dir / 'class_names.json'
    )
    
    # Тестируем на изображениях из валидации
    data_dir = Path(r"C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\IMG PREPROCESSING")
    
    # Берем по одному изображению из каждой папки
    test_images = []
    for class_name in predictor.class_names:
        class_dir = data_dir / class_name
        if class_dir.exists():
            img = next(class_dir.glob('*.jpg'), None)
            if img:
                test_images.append(img)
    
    print(f"\nТестируем на {len(test_images)} изображениях...")
    
    correct = 0
    total = 0
    
    for img_path in test_images[:10]:  # Первые 10
        true_class = img_path.parent.name
        predictions = predictor.predict(img_path)
        
        pred_class = predictions[0][0]
        pred_prob = predictions[0][1]
        
        is_correct = pred_class == true_class
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} {img_path.name[:50]}")
        print(f"   True: {true_class}")
        print(f"   Pred: {pred_class} ({pred_prob:.1f}%)")
        
        if len(predictions) > 1:
            print(f"   Top-3: {[(c, f'{p:.1f}%') for c, p in predictions]}")
    
    print(f"\n{'='*60}")
    print(f"Точность на тесте: {correct}/{total} = {100*correct/total:.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
