import os
import shutil
from pathlib import Path
import yaml

dataset_root = Path("C:/Users/GGamers/Desktop/FLC/hackhatons/lenta/dataset_color")

images_train = dataset_root / "images" / "train"
labels_train = dataset_root / "labels" / "train"

print("Удаление изображений без labels...")
img_names = set(f.stem for f in images_train.glob("*.jpg"))
lbl_names = set(f.stem for f in labels_train.glob("*.txt"))

missing_labels = img_names - lbl_names
print(f"   Найдено {len(missing_labels)} изображений без labels")

for name in missing_labels:
    img_path = images_train / f"{name}.jpg"
    if img_path.exists():
        img_path.unlink()

total_images = len(list(images_train.glob("*.jpg")))
total_labels = len(list(labels_train.glob("*.txt")))

print(f"\nИтого после очистки:")
print(f"   Изображений: {total_images}")
print(f"   Labels: {total_labels}")

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

print(f"\nСоздан data.yaml: {data_yaml_path}")

train_list_path = dataset_root / "train.txt"
with open(train_list_path, "w", encoding="utf-8") as f:
    for img in sorted(images_train.glob("*.jpg")):
        rel_path = img.relative_to(dataset_root)
        f.write(f"{rel_path}\n")

img_count = len(list(images_train.glob("*.jpg")))
print(f"Создан train.txt: {img_count} файлов")

print("\nДатасет готов к обучению!")
