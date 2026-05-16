import os
import shutil
from pathlib import Path
import yaml

dataset_root = Path("C:/Users/GGamers/Desktop/FLC/hackhatons/lenta/dataset_color")

images_train = dataset_root / "images" / "train"
labels_train = dataset_root / "labels" / "train"

images_train.mkdir(parents=True, exist_ok=True)
labels_train.mkdir(parents=True, exist_ok=True)

print("Copying images from video1-5...")
for i in range(1, 6):
    video_path = dataset_root / f"video{i}" / "train" / "images"
    if video_path.exists():
        for img in video_path.glob("*.jpg"):
            shutil.copy2(img, images_train / img.name)
        count = len(list(video_path.glob("*.jpg")))
        print(f"   video{i}: {count} images")

print("\nCopying labels from labels/1-5...")
for i in range(1, 6):
    labels_path = dataset_root / "labels" / str(i) / "labels" / "train"
    if labels_path.exists():
        for lbl in labels_path.glob("*.txt"):
            shutil.copy2(lbl, labels_train / lbl.name)
        count = len(list(labels_path.glob("*.txt")))
        print(f"   labels/{i}: {count} labels")

total_images = len(list(images_train.glob("*.jpg")))
total_labels = len(list(labels_train.glob("*.txt")))

print(f"\nTotal:")
print(f"   Images: {total_images}")
print(f"   Labels: {total_labels}")

if total_images != total_labels:
    print(f"\nWARNING: Image and label count mismatch!")
    img_names = set(f.stem for f in images_train.glob("*.jpg"))
    lbl_names = set(f.stem for f in labels_train.glob("*.txt"))
    missing_labels = img_names - lbl_names
    missing_images = lbl_names - img_names
    if missing_labels:
        print(f"   Missing labels for: {len(missing_labels)} images")
    if missing_images:
        print(f"   Missing images for: {len(missing_images)} labels")

data_yaml = {
    "path": str(dataset_root),
    "train": "images/train",
    "val": "images/train",
    "names": {
        0: "price_tag"
    }
}

data_yaml_path = dataset_root / "data.yaml"
with open(data_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"\nCreated data.yaml: {data_yaml_path}")

train_list_path = dataset_root / "train.txt"
with open(train_list_path, "w", encoding="utf-8") as f:
    for img in sorted(images_train.glob("*.jpg")):
        rel_path = img.relative_to(dataset_root)
        f.write(f"{rel_path}\n")

img_count = len(list(images_train.glob("*.jpg")))
print(f"Created train.txt: {img_count} files")

print("\nDataset ready for training!")
