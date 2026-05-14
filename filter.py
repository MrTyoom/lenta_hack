import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

MODEL_PATH = Path(__file__).parent / "epoch60.pt"
CONFIDENCE_THRESHOLD = 0.7


def draw_boxes(image, boxes, filtered=False):
    for box in boxes:
        conf = box.conf[0]
        if filtered and conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0) if filtered else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Фильтрация детекций по confidence")
    parser.add_argument("source", help="Путь к фотографии")
    args = parser.parse_args()

    model = YOLO(str(MODEL_PATH))

    image = cv2.imread(args.source)
    results = model(args.source)

    src = Path(args.source)

    img_no_filter = image.copy()
    draw_boxes(img_no_filter, results[0].boxes, filtered=False)
    cv2.imwrite(
        str(src.parent / (src.stem + "_without_filter.jpg")), img_no_filter)

    img_filtered = image.copy()
    draw_boxes(img_filtered, results[0].boxes, filtered=True)
    cv2.imwrite(str(src.parent / (src.stem + "_with_filter.jpg")), img_filtered)

    print("Готово! Смотри without_filter.jpg и with_filter.jpg")
