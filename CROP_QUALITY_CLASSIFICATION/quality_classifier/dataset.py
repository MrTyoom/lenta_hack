import random
from pathlib import Path

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader


def get_transforms(train=True, img_size=224):
    if train:
        return A.Compose([
            A.RandomResizedCrop((img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(p=0.4),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class QualityDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        if self.transform:
            image_np = self.transform(image=image_np)['image']
        return image_np, label, Path(img_path).name


def create_dataloaders(data_dir, batch_size=16, img_size=224, val_split=0.2):
    root = Path(data_dir)

    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_samples = []
    for class_name, idx in class_to_idx.items():
        class_dir = root / class_name
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in class_dir.glob(ext):
                all_samples.append((str(img_path), idx))

    print(f"Классы: {class_names}")
    print(f"Всего изображений: {len(all_samples)}")
    for name, idx in class_to_idx.items():
        count = sum(1 for _, l in all_samples if l == idx)
        print(f"  {name}: {count}")

    random.seed(42)
    random.shuffle(all_samples)
    split = int(len(all_samples) * (1 - val_split))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = QualityDataset(train_samples, class_to_idx, get_transforms(True, img_size))
    val_ds = QualityDataset(val_samples, class_to_idx, get_transforms(False, img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, class_names
