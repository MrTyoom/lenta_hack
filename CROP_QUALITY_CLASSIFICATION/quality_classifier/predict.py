import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import get_model


def load_predictor(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    cfg = checkpoint['config']
    class_names = checkpoint['class_names']

    model = get_model(cfg['model_name'], num_classes=cfg['num_classes'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    transform = A.Compose([
        A.Resize(cfg['img_size'], cfg['img_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return model, transform, class_names


def predict_image(model, transform, class_names, image):
    tensor = transform(image=image)['image'].unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_idx = probs.argmax().item()
    return class_names[top_idx], float(probs[top_idx]) * 100


def quality_classifier(model_path, crops):
    """
    crops: список кортежей [(track_id, crop_image), ...]
           где crop_image - numpy array в формате BGR (из OpenCV)
    возвращает: кортеж из 2 словарей
        - {track_id: is_trash} где True = 'bad'
        - {track_id: confidence} уверенность модели в %
    """
    model, transform, class_names = load_predictor(model_path)
    
    trash_map = {}
    confidence_map = {}
    
    for track_id, crop_bgr in crops:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        label, confidence = predict_image(model, transform, class_names, crop_rgb)
        trash_map[track_id] = (label == 'bad')
        confidence_map[track_id] = confidence
    
    return trash_map, confidence_map

# if __name__ == '__main__':
#     main()
