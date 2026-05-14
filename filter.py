import argparse
from pathlib import Path

import cv2
from paddleocr import PaddleOCR
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CONFIDENCE_THRESHOLD = 0.7


def primitive_filter(ocr_result):
    if not ocr_result or len(ocr_result) == 0:
        return False

    
    texts = ocr_result['rec_texts']

    nums = [t for t in texts if str(t).isdigit()]
    
    return (len(nums) >= 2) or ("9" in ''.join(nums))


def draw_boxes(ocr_engine, image, boxes, conf_threshold=0.7):
    crops = []
    meta = []

    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append(crop)
        meta.append((box, (x1, y1, x2, y2), conf))

    results = ocr_engine.ocr(crops)
    print(results, end='\n')

    filtered_meta = []

    for i, ocr_res in enumerate(results):
        if primitive_filter(ocr_res):
            filtered_meta.append(meta[i])

    for box, (x1, y1, x2, y2), conf in filtered_meta:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return image, filtered_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO + PaddleOCR фильтрация ценников"
    )
    parser.add_argument("source", help="Путь к изображению")
    args = parser.parse_args()

    # models
    model = YOLO(MODEL_PATH)
    ocr = PaddleOCR(
        lang="ru",
        use_angle_cls=True
    )

    image = cv2.imread(args.source)
    results = model(args.source)

    src = Path(args.source)

    img_no_filter = image.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(img_no_filter, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img_no_filter,
            f"{conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    cv2.imwrite(
        str(src.parent / (src.stem + "_without_filter.jpg")),
        img_no_filter
    )

    img_filtered = image.copy()

    _, filtered_boxes = draw_boxes(
        ocr,
        img_filtered,
        results[0].boxes,
        conf_threshold=CONFIDENCE_THRESHOLD
    )

    cv2.imwrite(
        str(src.parent / (src.stem + "_with_filter.jpg")),
        img_filtered
    )

    print("Готово! Смотри without_filter.jpg и with_filter.jpg")