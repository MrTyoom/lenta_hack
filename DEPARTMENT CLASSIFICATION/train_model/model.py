"""
Модели для классификации
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name='efficientnet-b0', num_classes=15, pretrained=True):
    """
    Создать модель для классификации
    
    Args:
        model_name: 'efficientnet-b0', 'efficientnet-b3', 'resnet-50', 'mobilenet-v3'
        num_classes: Количество классов
        pretrained: Использовать предобученные веса
    
    Returns:
        Модель PyTorch
    """
    
    if 'efficientnet' in model_name.lower():
        return create_efficientnet(model_name, num_classes, pretrained)
    elif 'resnet' in model_name.lower():
        return create_resnet(model_name, num_classes, pretrained)
    elif 'mobilenet' in model_name.lower():
        return create_mobilenet(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def create_efficientnet(model_name, num_classes, pretrained=True):
    """Создать EfficientNet"""
    
    from efficientnet_pytorch import EfficientNet
    
    model_name_map = {
        'efficientnet-b0': 'efficientnet-b0',
        'efficientnet-b3': 'efficientnet-b3',
        'efficientnet-b4': 'efficientnet-b4',
    }
    
    efficientnet_name = model_name_map.get(model_name.lower(), 'efficientnet-b0')
    
    if pretrained:
        model = EfficientNet.from_pretrained(efficientnet_name, num_classes=num_classes)
        print(f"Загружена {efficientnet_name} с предобученными весами (ImageNet)")
    else:
        model = EfficientNet.from_name(efficientnet_name, num_classes=num_classes)
        print(f"Создана {efficientnet_name} со случайными весами")
    
    return model


def create_resnet(model_name, num_classes, pretrained=True):
    """Создать ResNet"""
    
    model_name_map = {
        'resnet-18': models.resnet18,
        'resnet-34': models.resnet34,
        'resnet-50': models.resnet50,
        'resnet-101': models.resnet101,
    }
    
    resnet_name = model_name_map.get(model_name.lower(), 'resnet50')
    
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = resnet_name(weights=weights)
        print(f"Загружена ResNet-50 с предобученными весами (ImageNet)")
    else:
        model = resnet_name()
        print(f"Создана ResNet-50 со случайными весами")
    
    # Заменяем последний слой
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def create_mobilenet(model_name, num_classes, pretrained=True):
    """Создать MobileNetV3"""
    
    if pretrained:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        print(f"Загружена MobileNetV3 с предобученными весами (ImageNet)")
    else:
        model = models.mobilenet_v3_large()
        print(f"Создана MobileNetV3 со случайными весами")
    
    # Заменяем последний слой
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, num_classes)
    
    return model


def count_parameters(model):
    """Посчитать количество параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model):
    """Заморозить веса backbone"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем только последний слой
    if hasattr(model, '_fc'):
        for param in model._fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    print("Backbone заморожен, обучается только классификатор")


def unfreeze_model(model):
    """Разморозить все веса"""
    for param in model.parameters():
        param.requires_grad = True
    print("Все веса разморожены")


if __name__ == '__main__':
    # Тест
    model = get_model('efficientnet-b0', num_classes=15)
    print(f"Параметров: {count_parameters(model):,}")
    
    # Тестовый прогон
    x = torch.randn(4, 3, 224, 224)
    y = model(x)
    print(f"Вход: {x.shape}, Выход: {y.shape}")
