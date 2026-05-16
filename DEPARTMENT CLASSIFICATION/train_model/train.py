"""
Главный скрипт для обучения модели
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
import json
from datetime import datetime

from config import (
    DATA_DIR, MODEL_NAME, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, IMG_SIZE, DEVICE, SAVE_DIR
)
from dataset import create_dataloaders
from model import get_model, count_parameters


class Trainer:
    """Тренер для обучения модели"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss и optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        
        # Метрики
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self):
        """Обучение на одной эпохе"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Прогресс
            if (batch_idx + 1) % 10 == 0:
                acc = 100.0 * correct / total
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                      f"Loss={loss.item():.4f}, Acc={acc:.2f}%")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Валидация"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """Обучение модели"""
        print(f"\n{'='*60}")
        print(f"Начало обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Устройство: {self.device}")
        print(f"Эпох: {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 40)
            
            # Train
            print("Training...")
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            print("Validating...")
            val_loss, val_acc = self.validate()
            
            # Сохраняем в историю
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Вывод результатов
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Сохраняем лучшую модель
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth')
                print(f"  [BEST] Лучшая модель сохранена! (Acc: {val_acc:.2f}%)")
            
            # Сохраняем последнюю модель
            self.save_checkpoint('last_model.pth')
        
        print(f"\n{'='*60}")
        print(f"Обучение завершено!")
        print(f"Лучшая точность валидации: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return self.history
    
    def save_checkpoint(self, filename):
        """Сохранить чекпоинт"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'config': {
                'model_name': MODEL_NAME,
                'num_classes': NUM_CLASSES,
                'img_size': IMG_SIZE,
            }
        }
        
        save_path = SAVE_DIR / filename
        torch.save(checkpoint, save_path)
        print(f"  Сохранено: {save_path}")


def main():
    """Главная функция"""
    
    # Проверяем GPU
    if DEVICE == 'cuda' and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: GPU недоступен, используем CPU")
    
    # Создаем dataloaders
    print("\nЗагрузка данных...")
    train_loader, val_loader, class_names = create_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        test_split=0.2
    )
    
    print(f"\nКлассы: {class_names}")
    
    # Создаем модель
    print(f"\nСоздание модели: {MODEL_NAME}")
    model = get_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True)
    print(f"Параметров: {count_parameters(model):,}")
    
    # Создаем тренера
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE)
    
    # Обучаем
    history = trainer.train(NUM_EPOCHS)
    
    # Сохраняем историю
    history_path = SAVE_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nИстория сохранена: {history_path}")
    
    # Сохраняем классы
    classes_path = SAVE_DIR / 'class_names.json'
    with open(classes_path, 'w') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Классы сохранены: {classes_path}")
    
    print("\n✅ Обучение завершено!")


if __name__ == '__main__':
    main()
