"""
Dataset для загрузки изображений
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_transforms(train=True, img_size=224):
    """Получить аугментации"""
    
    if train:
        return A.Compose([
            A.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class DepartmentDataset(Dataset):
    """Dataset для классификации отделов"""
    
    def __init__(self, root_dir, class_to_idx, transform=None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        self.samples = []
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        
        # Собираем все изображения
        for class_name, class_idx in class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), class_idx))
        
        print(f"Загружено {len(self.samples)} изображений из {len(class_to_idx)} классов")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Открываем изображение
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Применяем трансформации
        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed['image']
        
        return image, label, Path(img_path).name
    
    def get_class_names(self):
        """Вернуть список имен классов"""
        return list(self.class_to_idx.keys())


def create_dataloaders(data_dir, batch_size=32, img_size=224, test_split=0.2):
    """Создать DataLoader для train и validation"""
    
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    import random
    
    root_dir = Path(data_dir)
    
    # Собираем все классы
    class_names = sorted([
        d.name for d in root_dir.iterdir() 
        if d.is_dir() and d.name not in {'annotation_tool', 'projects', '__pycache__', 'train_model'}
    ])
    
    print(f"Найдено классов: {len(class_names)}")
    print(f"Классы: {class_names}")
    
    # Создаем маппинг
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Собираем все сэмплы
    all_samples = []
    for class_name in class_names:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            continue
        
        for img_path in class_dir.glob('*.jpg'):
            all_samples.append((str(img_path), class_to_idx[class_name]))
    
    print(f"Всего изображений: {len(all_samples)}")
    
    # Разделяем на train/val
    random.seed(42)
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * (1 - test_split))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Создаем датасеты
    train_dataset = DepartmentDatasetWithSamples(
        train_samples, class_to_idx, 
        transform=get_transforms(train=True, img_size=img_size)
    )
    
    val_dataset = DepartmentDatasetWithSamples(
        val_samples, class_to_idx,
        transform=get_transforms(train=False, img_size=img_size)
    )
    
    # Создаем dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Для Windows лучше 0
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_names


class DepartmentDatasetWithSamples(Dataset):
    """Dataset с готовым списком сэмплов"""
    
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image_np)
            image = transformed['image']
        
        return image, label, Path(img_path).name
    
    def get_class_names(self):
        return list(self.class_to_idx.keys())
