import shutil
from pathlib import Path
import sys

source_folder = Path("C:/Users/GGamers/Desktop/FLC/hackhatons/lenta/data/photo")
output_folder = Path("C:/Users/GGamers/Desktop/FLC/hackhatons/lenta/data/photo_merged")

output_folder.mkdir(parents=True, exist_ok=True)

image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

print(f"Источник: {source_folder}")
print(f"Выход: {output_folder}\n")

counter = 0
for subfolder in source_folder.iterdir():
    if not subfolder.is_dir():
        continue
    
    folder_name = subfolder.name
    print(f"Обработка папки: {folder_name}")
    
    for img_file in subfolder.iterdir():
        if img_file.suffix.lower() in image_extensions:
            counter += 1
            new_name = f"{folder_name}_img_{counter:06d}{img_file.suffix}"
            new_path = output_folder / new_name
            
            shutil.copy2(str(img_file), str(new_path))
            print(f"  {img_file.name} -> {new_name}")
    
    print(f"  Обработано: {counter}\n")

print(f"\n✅ Готово! Всего файлов: {counter}")
print(f"Папка: {output_folder}")
