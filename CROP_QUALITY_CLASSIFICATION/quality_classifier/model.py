import torch.nn as nn
from torchvision import models


def get_model(model_name='mobilenet-v3', num_classes=3, pretrained=True):
    if 'mobilenet' in model_name.lower():
        return _mobilenet(num_classes, pretrained)
    elif 'efficientnet' in model_name.lower():
        return _efficientnet(model_name, num_classes, pretrained)
    elif 'resnet' in model_name.lower():
        return _resnet(num_classes, pretrained)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def _mobilenet(num_classes, pretrained):
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _efficientnet(model_name, num_classes, pretrained):
    from efficientnet_pytorch import EfficientNet
    name = 'efficientnet-b0'
    if 'b3' in model_name:
        name = 'efficientnet-b3'
    if pretrained:
        model = EfficientNet.from_pretrained(name, num_classes=num_classes)
    else:
        model = EfficientNet.from_name(name, num_classes=num_classes)
    return model


def _resnet(num_classes, pretrained):
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
