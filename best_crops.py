import cv2
import torch
import time
import numpy as np

from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


MODEL_PATH = "models/best.pt"
INPUT_FOLDER = "data/video1/25_12-20"
OUTPUT_FOLDER = "data/tmp/obj_det"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

TOP_K = 3

MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 20
SHARPNESS_THRESHOLD = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

model = YOLO(MODEL_PATH).to(DEVICE)

video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

video_files = [
    f for f in Path(INPUT_FOLDER).iterdir()
    if f.suffix.lower() in video_extensions and not f.name.startswith("~")
]


def compute_crop_quality(crop, conf):
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]

    if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    if sharpness < SHARPNESS_THRESHOLD:
        return None

    area = w * h
    aspect_ratio = w / (h + 1e-6)

    aspect_bonus = 1.0 if 1.2 <= aspect_ratio <= 6.0 else 0.7

    area_score = np.sqrt(area)

    score = (
        sharpness * 0.5 +
        area_score * 0.3 +
        conf * 100 * 0.2
    ) * aspect_bonus

    return float(score)


global_start = time.time()

for video_path in video_files:

    print(f"\nProcessing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    best_crops = defaultdict(list)

    frame_count = 0
    total_detections = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, conf, track_id in zip(boxes, confs, track_ids):

                x1, y1, x2, y2 = map(int, box)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                crop = frame[y1:y2, x1:x2]

                score = compute_crop_quality(crop, conf)

                if score is None:
                    continue

                best_crops[track_id].append({
                    "score": score,
                    "crop": crop.copy(),
                    "frame": frame_count,
                    "conf": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })

                best_crops[track_id] = sorted(
                    best_crops[track_id],
                    key=lambda x: x["score"],
                    reverse=True
                )[:TOP_K]

                total_detections += 1

        if frame_count % 100 == 0:
            print(f"Frame {frame_count} | detections {total_detections}")

    cap.release()



    out_dir = Path(OUTPUT_FOLDER) / "best_crops" / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for track_id, items in best_crops.items():

        track_dir = out_dir / f"id_{track_id}"
        track_dir.mkdir(exist_ok=True)

        for i, item in enumerate(items):

            path = track_dir / (
                f"top{i+1}"
                f"_frame{item['frame']}"
                f"_conf{item['conf']:.2f}"
                f"_score{item['score']:.1f}.jpg"
            )

            cv2.imwrite(str(path), item["crop"])

    print(f"Saved crops for {video_path.name}")

print(f"\nDone in {(time.time() - global_start)/60:.2f} min")