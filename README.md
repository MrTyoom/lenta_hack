# Lenta Hackathon Project

## Структура проекта

```
lenta/
├── .gitignore                    # Игнорируемые файлы
├── README.md                     # Документация
├── requirements.txt              # Зависимости Python
│
├── venv/                         # Виртуальное окружение (игнорируется)
│
├── DEPARTMENT CLASSIFICATION/    # Классификация отделов
│   ├── department/               # Изображения отделов (игнорируется)
│   └── train_model/              # Обучение модели
│       ├── models/               # Чекпоинты (игнорируется)
│       ├── model.py              # Модели EfficientNet/ResNet
│       ├── train.py              # Скрипт обучения
│       ├── predict_single.py     # Предсказание для фото
│       ├── dataset.py            # Датасет и аугментации
│       └── config.py             # Конфигурация
│
├── YOLO/                         # Детекция ценников
│   ├── data/                     # Датасеты (игнорируется)
│   ├── runs/                     # Результаты обучения (игнорируется)
│   ├── video_output/             # Обработанные видео (игнорируется)
│   ├── train_yolo.py             # Обучение YOLO
│   └── test_yolo.py              # Тестирование YOLO
│
├── TEXT RECOGNIZER/              # Распознавание текста
│   ├── create_bm25.py            # Создание BM25 индекса
│   ├── hybrid_search.py          # Поиск по продуктам
│   ├── init_search.py            # Инициализация поиска
│   └── test_search.py            # Тестирование поиска
│
├── LLMTEXT/                      # Текстовая обработка
│   ├── bge-m3/                   # Модель (игнорируется)
│   └── ...                       # Скрипты работы с текстом
│
├── VLM TEXT RECOGNIZER/          # VLM для OCR
│   └── AVITO/                    # Данные (игнорируется)
│
└── TRAKING BBOX VIDEO AND PHOTO/ # Трекинг объектов
    ├── photo_track.py            # Обработка фото
    └── video_track.py            # Обработка видео
```

## Что в git / что игнорируется

### ✅ В git
- `.py` скрипты
- `.md` документация
- `.json` конфиги
- `.ipynb` ноутбуки
- `.txt` файлы зависимостей

### ❌ Игнорируется
- Медиа файлы (`*.jpg`, `*.png`, `*.mp4`, `*.avi`)
- Данные (`data/`, `IMG PREPROCESSING/`, `YOLO/data/`)
- Модели и чекпоинты (`models/`, `runs/`, `bge-m3/`)
- Результаты (`*.csv`, `*.xlsx`, `*.db`, `*.pkl`)
- Виртуальное окружение (`venv/`)
- Кэш (`__pycache__/`, `*.pyc`)

---

## Требования

### Аппаратные
- GPU: NVIDIA с поддержкой CUDA (рекомендуется RTX 40/50 серия)
- Минимум 16GB RAM
- 50GB свободного места на диске

### Программные
- Windows 10/11
- NVIDIA Driver (версия 591.86 или новее)
- Python 3.12
- WSL2 с Ubuntu (для vllm)

---

## Установка

### 1. Создание виртуального окружения

```powershell
# Создание venv
python -m venv venv

# Активация
.\venv\Scripts\Activate.ps1
```

### 2. Установка PyTorch с CUDA поддержкой

**Для RTX 40/50 серии (Blackwell архитектура):**
```powershell
# Nightly версия с CUDA 12.8 (требуется для RTX 5070)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Для других GPU:**
```powershell
# CUDA 12.4 (RTX 30/40 серия)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (старые GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU только
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Установка зависимостей

```powershell
pip install ultralytics opencv-python
```

### 4. Проверка установки

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

---

## Запуск проектов

### Video Tracking (YOLO)

```powershell
.\venv\Scripts\Activate.ps1
python .\video_track.py
```

**Настройки в video_track.py:**
- `MODEL_PATH` - путь к весах YOLO
- `INPUT_FOLDER` - папка с видео
- `OUTPUT_FOLDER` - папка для результатов
- `CONF_THRESHOLD` - порог уверенности (0.25)
- `IOU_THRESHOLD` - порог IoU (0.45)
- `ROTATE` - поворот видео (True/False)

---

### VLM Tests (vllm в WSL2)

**Требования:** WSL2 с Ubuntu

**Установка vllm в WSL2:**
```bash
# В Ubuntu (WSL2)
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --break-system-packages
~/.local/bin/pip install vllm --break-system-packages
```

**Запуск vllm сервера (2 терминала):**

Терминал 1 - запуск сервера:
```bash
# Запуск сервера (займет 2-5 минут на загрузку модели)
~/lenta_hack/start_vllm.sh

# Или вручную:
~/.local/bin/vllm serve OpenGVLab/InternVL3_5-8B \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.72 \
  --enforce-eager
```

Дождитесь сообщения: `Uvicorn running on http://0.0.0.0:8000`

Терминал 2 - запуск тестов:
```bash
# Копирование данных
mkdir -p ~/lenta_hack/data
cp /mnt/c/Users/GGamers/Desktop/FLC/hackhatons/lenta/data/25_12-20/annotations/images/25_12-20_frame_000005.jpg ~/lenta_hack/data/crop1.jpg

# Запуск теста
cd ~/lenta_hack
python3 test_vlm.py
```

**Запуск ноутбука:**
```bash
# Установка jupyter (если нет)
~/.local/bin/pip install jupyter notebook --break-system-packages

# Запуск
cd ~/lenta_hack
jupyter notebook vlm_tests.ipynb
```

---

## Решение проблем

### Ошибка: Файлы данных отсутствуют
Все данные игнорируются git. Восстановите их заново:
```powershell
# Для Department Classification
python DEPARTMENT CLASSIFICATION/train_model/train.py

# Для YOLO
python YOLO/train_yolo.py

# Для TEXT RECOGNIZER
python TEXT RECOGNIZER/init_search.py
```

### Ошибка: `ModuleNotFoundError: No module named 'vllm._C'`
vllm не работает на Windows нативно. Используйте WSL2.

### Ошибка: `CUDA available: False`
Убедитесь, что установлена CUDA-версия PyTorch, а не CPU:
```powershell
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Ошибка: `sm_XXX is not compatible`
Ваша видеокарта слишком новая для стабильной версии PyTorch. Используйте nightly-версию с CUDA 12.8.

### Ошибка: `torchvision::nms does not exist`
Несовместимость версий torch и torchvision. Переустановите оба пакета:
```powershell
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Полезные команды

**Проверка версий:**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
nvcc --version
nvidia-smi
```

**Очистка кэша pip:**
```powershell
pip cache purge
```

**Полная переустановка:**
```powershell
pip uninstall torch torchvision torchaudio ultralytics -y
pip cache purge
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics
```

**Git команды:**
```powershell
# Проверка статуса
git status

# Что будет закоммичено
git status --short

# Отмена изменений в игнорируемых файлах
git clean -fdx
```

---

## Структура проекта

```
lenta/
├── venv/                    # Виртуальное окружение
├── video_track.py           # Скрипт обработки видео
├── README.md                # Эта инструкция
├── data/
│   ├── Unlabeled/           # Входные видео
│   └── 25_12-20/            # Данные для VLM
│       └── annotations/
│           └── images/      # Изображения для OCR
├── video_output/            # Результаты обработки видео
└── runs/detect/             # YOLO модели
    └── yolo_training/
        └── price_tag_yolo26s/
            └── weights/
                └── best.pt  # Веса модели
```
