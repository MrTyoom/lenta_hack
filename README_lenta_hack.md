# Lenta Hackathon — Система инспекции ценников

Система компьютерного зрения для автоматической обработки видео с торговых залов: обнаружение ценников, оценка качества, OCR и сопоставление с каталогом товаров.

---

## Быстрый старт

```bash
# 1. Клонировать
git clone https://github.com/MrTyoom/lenta_hack.git
cd lenta_hack
git checkout yolo_detect

# 2. Подтянуть модели (Git LFS) — обязательно!
git lfs pull

# 3. Окружение и зависимости
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. Скачать HF-модели (~15 ГБ, один раз)
python setup_models.py

# 5. Запустить
jupyter notebook main1.ipynb
# или
cd backend_streamlit && streamlit run app.py
```

---

## Запуск

### Вариант 1 — Jupyter Notebook (рекомендуется для разработки)

```bash
jupyter notebook main1.ipynb
```

Основной файл проекта — `main1.ipynb`. Ноутбук выполняет полный пайплайн:
1. Загрузка YOLO, EfficientNet (отдел), MobileNetV3 (качество)
2. Детекция и трекинг ценников на видео
3. Классификация отдела по полноэкранным кадрам
4. Классификация цвета ценника (красный/жёлтый/белый)
5. Классификация качества кропа (мусор/нормальный)
6. Удаление дубликатов по косинусному сходству
7. VLM OCR (AvitoTech/a-vision) — распознавание текста с кропов
8. Поиск товара в каталоге (FAISS + BM25 + Levenshtein)
9. LLM-валидация (Qwen2.5-7B-Instruct) — выбор SKU из топ-5

### Вариант 2 — Streamlit веб-интерфейс

```bash
cd backend_streamlit
streamlit run app.py
```

Открывается `http://localhost:8501`:
- Загрузка видео (mp4, avi, mov)
- Запуск пайплайна обработки
- Просмотр распознанных ценников
- Скачивание результатов в CSV

---

## Требования

### Железо
- GPU NVIDIA с CUDA (RTX 30/40/50 серии)
- ОЗУ: минимум 16 ГБ
- Диск: ~50 ГБ (из них ~15 ГБ — HF-модели)

### ПО
- Python 3.12+
- CUDA-драйвер NVIDIA
- Git LFS

### Зависимости

Основные (полный список в `requirements.txt`):

```
torch torchvision torchaudio  # PyTorch с CUDA
ultralytics                   # YOLOv8
transformers accelerate       # HF-модели (Qwen, AvitoTech)
bitsandbytes                  # 4-bit/8-bit квантование
sentence-transformers faiss-cpu  # Эмбеддинги и поиск
rapidfuzz python-Levenshtein  # Fuzzy-матчинг
albumentations opencv-python-headless  # Изображения
efficientnet_pytorch          # EfficientNet
qwen-vl-utils                 # VLM-утилиты
streamlit                     # Веб-интерфейс
```

---

## Данные и модели

### Что уже в репозитории (скачивается через `git clone` + `git lfs pull`)

| Файл | Размер | Назначение |
|------|--------|------------|
| `MAIN_MODULE/models/best.pt` | 19 МБ | YOLOv8 — детекция ценников |
| `DEPARTMENT_CLASSIFICATION/train_model/models/best_model.pth` | 47 МБ | EfficientNet-B0 — классификация 15 отделов |
| `CROP_QUALITY_CLASSIFICATION/quality_classifier/models/best_model.pth` | 16 МБ | MobileNetV3 — классификация качества кропов |
| `match.xlsx` | 19 МБ | Таблица матчинга (data ↔ db_hack, 50k+625k строк) |
| `LLMTEXT/site_total.xlsx` | 5.5 МБ | Каталог товаров для SKU-поиска |
| `LLMTEXT/db_hack.csv` | 48 МБ | База товаров для матчинга |

### Что скачивается с HuggingFace (`python setup_models.py`)

| Модель | HF ID | Размер | Назначение |
|--------|-------|--------|------------|
| USER-bge-m3 | `deepvk/USER-bge-m3` | ~1.4 ГБ | Эмбеддинги для семантического поиска |
| A-Vision | `AvitoTech/a-vision` | ~14 ГБ | VLM для OCR с ценников (Qwen2.5-VL-7B) |

### Что скачивается автоматически при первом использовании

| Модель | HF ID | Размер | Назначение |
|--------|-------|--------|------------|
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | ~5 ГБ (4-bit) | LLM-валидация SKU (в ноутбуке) |

---

## Параметры (`params.yaml`)

```yaml
is_test: true  # true — тестовый режим (ограниченный вывод)

main_extraction:
  input_folder: "data/25_12-20"       # папка с видеофайлами
  output_folder: "data/extracted_crops"
  model_path: "MAIN_MODULE/models/best.pt"
  min_crop_width: 50
  min_crop_height: 50
  sharpness_threshold: 50.0
  top_k: 1
  conf_threshold: 0.25
  iou_threshold: 0.45
  rotate_frames: true
  tracker_config: "bytetrack.yaml"

department_classifier:
  model_path: "DEPARTMENT_CLASSIFICATION/train_model/models/best_model.pth"

quality_classifier:
  model_path: "CROP_QUALITY_CLASSIFICATION/quality_classifier/models/best_model.pth"
```

Положите видеофайлы в `data/25_12-20/` перед запуском.

---

## Поиск товаров по тексту (LLMTEXT)

Инициализация индекса (один раз, создаёт `lenta_products.db` и FAISS-индекс):
```bash
cd LLMTEXT
python init_search.py
```

Пакетный поиск по JSONL:
```bash
python batch_search.py --input data.jsonl --output results.jsonl
```

Формат `data.jsonl`:
```json
{"id": "1", "ocr_text": "Мед БЕРЕСТОВ 500г натуральный"}
{"id": "2", "ocr_text": "Молоко Простоквашино 3.2% 1л"}
```

---

## Дообучение моделей

### Классификатор отдела (EfficientNet-B0)

```bash
cd DEPARTMENT_CLASSIFICATION/train_model
python train.py
```

### Классификатор качества (MobileNetV3)

```bash
cd CROP_QUALITY_CLASSIFICATION/quality_classifier
python train.py
```

После обучения `best_model.pth` и `class_names.json` появятся в папке `models/`.

---

## Типичные проблемы

| Проблема | Решение |
|---|---|
| `torch.cuda.is_available()` → `False` | Переустановить PyTorch с CUDA-суффиксом (`cu121` для RTX 40/50) |
| `.pt`/`.pth` файлы пустые (0 байт) | Выполнить `git lfs pull` |
| `ModuleNotFoundError: X` | `pip install -r requirements.txt` |
| Ошибка памяти GPU | Уменьшить batch_size, использовать 4-bit конфиг VLM |
| VLM падает с `Int8Params` | Использовать конфиг `4-bit NF4` вместо 8-bit |
| `faiss` не устанавливается | Ставится `faiss-cpu` (из `requirements.txt`) |

---

## Архитектура проекта

```
lenta_hack/
├── MAIN_MODULE/               # YOLO-детекция + трекинг + OCR
│   ├── src/
│   │   ├── main.py
│   │   ├── crop_extraction.py
│   │   ├── distortion.py
│   │   └── ocr_processing.py   # PaddleOCR (не используется в main1.ipynb)
│   └── models/best.pt         # YOLO-веса (в LFS)
│
├── CROP_QUALITY_CLASSIFICATION/  # MobileNetV3: качество кропов
│   └── quality_classifier/
│       ├── train.py / predict.py
│       └── models/best_model.pth
│
├── DEPARTMENT_CLASSIFICATION/    # EfficientNet-B0: 15 отделов
│   └── train_model/
│       ├── train.py / predict_single.py
│       └── models/best_model.pth
│
├── LLMTEXT/                  # Поиск товаров по тексту
│   ├── init_search.py        # Инициализация FAISS + BM25
│   ├── product_matcher.py    # Топ-5 матчинг
│   ├── hf_sku_matcher.py     # LLM-валидация (Qwen2.5-7B)
│   └── batch_search.py       # Пакетный поиск
│
├── VLM_MODULE/               # VLM OCR (AvitoTech/a-vision)
│   └── detect.py
│
├── IMG PREPROCESSING/        # Классификация цвета ценника
│   └── color_classifier.py
│
├── backend_streamlit/        # Веб-интерфейс
│   └── app.py
│
├── VRAM_CLEAN/               # Управление VRAM
│   └── vram_cleanup.py
│
├── params.yaml               # Единый конфиг
├── setup_models.py           # Скачивание HF-моделей
├── requirements.txt          # Python-зависимости
├── main1.ipynb               # Основной Jupyter-ноутбук
├── main.ipynb                # Черновой ноутбук
└── match_v3.py               # Матчинг data ↔ db_hack
```

---

## Стек технологий

| Компонент | Технология |
|---|---|
| Детекция | YOLOv8 (Ultralytics) + ByteTrack |
| Качество кропов | MobileNetV3-Large (PyTorch) |
| Отдел магазина | EfficientNet-B0 (PyTorch) |
| OCR | AvitoTech/a-vision (VLM) |
| Эмбеддинги | deepvk/USER-bge-m3 (SentenceTransformer) |
| Поиск | FAISS HNSW + BM25 + Levenshtein |
| LLM-валидация | Qwen2.5-7B-Instruct (4-bit) |
| Веб-интерфейс | Streamlit |
| Конфигурация | OmegaConf / YAML |
