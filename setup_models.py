"""
Скачивание HF-моделей при первом запуске проекта.

Запуск:
    python setup_models.py

Скачивает:
    1. deepvk/USER-bge-m3 (~1.4 ГБ) -> LLMTEXT/bge-m3/
    2. AvitoTech/a-vision    (~14 ГБ) -> VLM_MODULE/AVITO/

Кастомные модели (best.pt, best_model.pth) должны лежать в репозитории.
"""
import os, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()


def download_sentence_transformer(model_id, save_dir):
    """Скачать SentenceTransformer модель и сохранить локально."""
    from sentence_transformers import SentenceTransformer

    save_path = SCRIPT_DIR / save_dir
    if save_path.exists():
        print(f"[OK] Уже скачана: {save_path}")
        return

    print(f"[>>] Скачивание {model_id}...")
    model = SentenceTransformer(model_id)
    model.save(str(save_path))
    print(f"[OK] Сохранена: {save_path}")


def download_hf_model(model_id, save_dir, model_cls_name="AutoModelForImageTextToText"):
    """Скачать HF-модель и сохранить локально."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    save_path = SCRIPT_DIR / save_dir
    if save_path.exists():
        print(f"[OK] Уже скачана: {save_path}")
        return

    print(f"[>>] Скачивание {model_id} (это большая модель, ждите)...")
    model = AutoModelForImageTextToText.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))
    print(f"[OK] Сохранена: {save_path}")


def check_custom_models():
    """Проверить наличие кастомных моделей в репозитории."""
    models = {
        "MAIN_MODULE/models/best.pt": "YOLO (обнаружение ценников)",
        "DEPARTMENT_CLASSIFICATION/train_model/models/best_model.pth": "EfficientNet-B0 (классификация отделов)",
        "DEPARTMENT_CLASSIFICATION/train_model/models/class_names.json": "Имена классов отделов",
        "CROP_QUALITY_CLASSIFICATION/quality_classifier/models/best_model.pth": "MobileNetV3 (качество кропов)",
    }
    all_ok = True
    for rel_path, desc in models.items():
        full = SCRIPT_DIR / rel_path
        if full.exists():
            print(f"[OK] {desc}: {rel_path}")
        else:
            print(f"[!!] НЕ НАЙДЕНА: {desc} -> {rel_path}")
            all_ok = False

    if not all_ok:
        print("\n[!!] Некоторые кастомные модели отсутствуют!")
        print("[!!] Убедитесь, что вы склонировали репозиторий с Git LFS.")
        print("[!!] Команда: git lfs pull")
    else:
        print("[OK] Все кастомные модели на месте.")


def main():
    print("=" * 60)
    print("SETUP MODELS — проверка и скачивание")
    print("=" * 60)

    # 1. Проверка кастомных моделей
    print("\n[1/3] Проверка кастомных моделей...")
    check_custom_models()

    # 2. Скачивание embedding-модели
    print("\n[2/3] Скачивание embedding-модели (USER-bge-m3)...")
    download_sentence_transformer("deepvk/USER-bge-m3", "LLMTEXT/bge-m3")

    # 3. Скачивание VLM-модели
    print("\n[3/3] Скачивание VLM-модели (AvitoTech/a-vision)...")
    print("[i]  Это ~14 ГБ. Если уже скачана — пропустится.")
    download_hf_model("AvitoTech/a-vision", "VLM_MODULE/AVITO")

    print("\n" + "=" * 60)
    print("ГОТОВО. Можно запускать main.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
