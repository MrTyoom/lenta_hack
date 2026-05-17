from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='../data/data.yaml',
    epochs=100,
    imgsz=1280, 
    batch=4, # экспериментируем
    optimizer='Adam',
    lr0='0.001',
    device='0',
    name='price_tag',
    save=True,
    save_period=10,
    save_json='True',
    project='runs',
    exist_ok=True,
    resume=False,
    patience=10,
    rect=True
)