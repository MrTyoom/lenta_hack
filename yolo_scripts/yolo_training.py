from ultralytics import YOLO

model = YOLO("YOUR_PATH") # путь, куда сохранится обученная модель

metrics = model.val(
    data='../data/data.yaml',
    split='val',
    imgsz=1280,
    conf=0.9,
    iou=0.6,
    device='0'
)