# 🔖 Авто-разметка и обучение YOLO для детекции ценников

## 📋 Описание

Автоматическая подготовка датасета и обучение YOLOv8n для детекции ценников на изображениях и видео.

**Используемые технологии:**
- **Grounding DINO** — zero-shot детекция для авто-разметки
- **YOLOv8n** — быстрая и точная детекция после обучения
- **CUDA** — ускорение на GPU (RTX 5070)

---

## 🚀 Быстрый старт

### 1️⃣ Разметка фотографий (авто-разметка)

#### Вариант A: Grounding DINO (без обучения)
```bash
python main.py
```
- **Вход:** `photo/`
- **Выход:** `color_filtered_output/` (фото с боксами + detections.csv)
- Использует цветовой фильтр + Grounding DINO
- Создаёт DataFrame: image, box_number, x, y, w, h, area, confidence

#### Вариант B: YOLO модель (после обучения)
```bash
python photo_track.py
```
- **Вход:** `photo/`
- **Выход:** `photo_output/` (фото с боксами + detections.csv)
- Использует обученную YOLO модель
- Создаёт DataFrame: image, box_number, x1, y1, x2, y2, w, h, area, confidence, class

---

### 2️⃣ Разметка изображений (для обучения YOLO)

```bash
python prepare_dataset.py "путь\к\папке\с\фото" --color-filter --output_folder dataset
```

**Пример:**
```bash
python prepare_dataset.py "C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\photo" --threshold 0.2 --text-threshold 0.15 --color-filter
```

**Параметры:**
| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--threshold` | 0.2 | Box threshold (ниже = больше детекций) |
| `--text-threshold` | 0.15 | Text threshold |
| `--color-filter` | выключен | Фильтр по цвету (белый/красный/желтый) |
| `--min-area` | 500 | Мин. площадь детекции |
| `--train-split` | 0.8 | Доля train (80% train, 20% val) |
| `--output_folder` | dataset | Папка для датасета |

**Результат:**
```
dataset/
├── data.yaml                 # Конфиг для YOLO
├── images/
│   ├── train/               # Изображения для обучения
│   └── val/                 # Изображения для валидации
└── labels/
    ├── train/               # YOLO аннотации (.txt)
    └── val/
```

---

### 3️⃣ Разметка видео (для обучения YOLO)

---

### 3️⃣ Разметка видео (для обучения YOLO)

#### Вариант A: Grounding DINO (авто-разметка для обучения)
```bash
python prepare_dataset.py "путь\к\папке\с\видео" --interval 5 --color-filter
```

**Пример:**
```bash
python prepare_dataset.py "C:\Users\GGamers\Desktop\FLC\hackhatons\lenta\data\Unlabeled" --interval 5 --threshold 0.2 --text-threshold 0.15 --color-filter
```

**Параметры:**
| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--interval` | 5 | Каждый N-й кадр из видео |
| `--threshold` | 0.2 | Box threshold (ниже = больше детекций) |
| `--text-threshold` | 0.15 | Text threshold |
| `--color-filter` | выключен | Фильтр по цвету (белый/красный/желтый) |
| `--min-area` | 500 | Мин. площадь детекции |
| `--train-split` | 0.8 | Доля train (80% train, 20% val) |
| `--output_folder` | dataset | Папка для датасета |

**Результат:**
```
dataset/
├── data.yaml                 # Конфиг для YOLO
├── images/
│   ├── train/               # Изображения для обучения
│   └── val/                 # Изображения для валидации
└── labels/
    ├── train/               # YOLO аннотации (.txt)
    └── val/
```

#### Вариант B: YOLO модель (разметка готовых видео)
```bash
python video_track.py
```
- **Вход:** `data/Unlabeled/`
- **Выход:** `video_output/` (видео с боксами)
- Поворот: 90° против часовой
- Детекция: каждые 5 кадров
- Сохраняет время обработки

#### Вариант C: YOLO модель (без поворота)
```bash
python video_output1_track.py
```
- **Вход:** `video_output1/`
- **Выход:** `video_output1/` (видео с боксами + detections.csv)
- **Поворот:** отключён (можно включить в настройках)
- Детекция: каждые 5 кадров
- Создаёт DataFrame: video, frame, x1, y1, x2, y2, w, h, area, confidence, class

**Настройки в файле:**
```python
ROTATE = False  # True = 90° против часовой
DETECT_EVERY = 5  # Детекция каждые N кадров
```

---

### 4️⃣ Обучение YOLO

```bash
python train_yolo.py dataset --epochs 100 --batch 16
```

**Параметры:**
| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--epochs` | 100 | Количество эпох обучения |
| `--batch` | 16 | Размер батча |
| `--imgsz` | 640 | Размер изображения |
| `--device` | 0 | GPU device |
| `--name` | price_tag_yolo8n | Название эксперимента |

**Результат:**
```
yolo_training/price_tag_yolo8n/
├── weights/
│   ├── best.pt          # Лучшая модель
│   └── last.pt          # Последняя модель
└── results.csv          # Метрики обучения
```

---

### 5️⃣ Тестирование модели

```bash
# Веб-камера
python test_yolo.py yolo_training/price_tag_yolo8n/weights/best.pt --source 0

# Изображение
python test_yolo.py yolo_training/price_tag_yolo8n/weights/best.pt --source "image.jpg" --save

# Видео
python test_yolo.py yolo_training/price_tag_yolo8n/weights/best.pt --source "video.mp4" --save
```

---

## 📁 Пример полного цикла

### Для фото (разметка для обучения):
```bash
# 1. Авто-разметка фото через Grounding DINO (создаёт YOLO датасет)
python prepare_dataset.py "photo" --color-filter --output_folder dataset_photo

# 2. Обучение модели
python train_yolo.py dataset_photo --epochs 100 --batch 16

# 3. Тест на фото через YOLO
python photo_track.py
```

### Для фото (быстрая разметка без обучения):
```bash
# 1. Авто-разметка фото через YOLO
python photo_track.py

# 2. Проверка результатов в photo_output/detections.csv
```

### Для видео с обучением:
```bash
# 1. Подготовка датасета из видео ИЛИ фото
python prepare_dataset.py "data\Unlabeled" --interval 5 --color-filter --output_folder dataset

# 2. Обучение (100 эпох)
python train_yolo.py dataset --epochs 100 --batch 16 --name price_tags_v1

# 3. Разметка видео обученной моделью
python video_track.py

# 4. Тест на веб-камере
python test_yolo.py yolo_training/price_tags_v1/weights/best.pt --source 0
```

### Для изображений с обучением:
```bash
# 1. Подготовка датасета из изображений
python prepare_dataset.py "photo" --color-filter --output_folder dataset

# 2. Обучение модели
python train_yolo.py dataset --epochs 100 --batch 16 --name price_tags_v1

# 3. Разметка фото обученной моделью
python photo_track.py
```

### Для готовых видео (без обучения):
```bash
# 1. Разметка видео (без поворота)
python video_output1_track.py

# 2. Проверка результатов в video_output1/detections.csv
```

---

## ⚙️ Настройка параметров

### Если датасет маленький (< 100 изображений):
```bash
# Подготовка - каждый кадр (для видео) ИЛИ все фото (для изображений)
python prepare_dataset.py "data\videos" --interval 1 --color-filter
python prepare_dataset.py "photo" --color-filter

# Обучение - больше эпох
python train_yolo.py dataset --epochs 200 --batch 8
```

### Если датасет большой (> 1000 изображений):
```bash
# Подготовка - каждый 10-й кадр (для видео)
python prepare_dataset.py "data\videos" --interval 10

# Для изображений - уменьшить пороги детекции
python prepare_dataset.py "photo" --threshold 0.3 --min-area 800
```

### Если много ложных срабатываний:
```bash
# Подготовка - выше пороги (для видео и фото)
python prepare_dataset.py "data\videos" --threshold 0.3 --text-threshold 0.25 --min-area 800
python prepare_dataset.py "photo" --threshold 0.3 --min-area 800
```

### Если пропускает ценники:
```bash
# Подготовка - ниже пороги (для видео и фото)
python prepare_dataset.py "data\videos" --threshold 0.15 --text-threshold 0.1 --min-area 300
python prepare_dataset.py "photo" --threshold 0.15 --min-area 300
```

---

## 📊 Формат данных

### prepare_dataset.py (для обучения YOLO)

**Вход:** Видео ИЛИ изображения  
**Выход:** YOLO-датасет с аннотациями (.txt)

```
dataset/
├── data.yaml                 # Конфиг для YOLO
├── images/
│   ├── train/               # Изображения для обучения
│   └── val/                 # Изображения для валидации
└── labels/
    ├── train/               # YOLO аннотации (.txt)
    └── val/
```

### data.yaml
```yaml
train: C:\path\to\dataset\images\train
val: C:\path\to\dataset\images\val

nc: 1
names:
  - price_tag
```

### YOLO аннотации (.txt)
Каждая строка: `class_id x_center y_center width height`

Пример:
```
0 0.453125 0.618750 0.039844 0.256944
0 0.608203 0.614583 0.044531 0.279167
```

### detections.csv (для фото и видео)
| Колонка | Описание |
|---------|----------|
| `image` / `video` | Имя файла |
| `box_number` / `frame` | Номер бокса или кадр |
| `x1, y1, x2, y2` | Координаты углов |
| `w, h` | Ширина и высота |
| `area` | Площадь (w × h) |
| `confidence` | Вероятность детекции |
| `class` | Класс объекта |

---

## 🎯 Рекомендации по обучению

### Минимальный датасет:
- **50-100 изображений** — базовое качество
- **200-500 изображений** — хорошее качество
- **1000+ изображений** — отличное качество

### Параметры обучения:
| Размер датасета | Epochs | Batch | Время (RTX 5070) |
|-----------------|--------|-------|------------------|
| 50-100 | 200 | 8 | ~10 мин |
| 200-500 | 100 | 16 | ~30 мин |
| 1000+ | 50 | 32 | ~1 час |

### Метрики качества:
После обучения проверьте:
- **mAP50** > 0.8 — отлично
- **mAP50** > 0.6 — хорошо
- **mAP50** < 0.5 — нужно больше данных или эпох

---

## 🔧 Требования

- Python 3.10+
- torch >= 2.0 (с CUDA)
- ultralytics >= 8.0
- transformers >= 4.35
- opencv-python
- scikit-learn
- Pillow
- pandas

### Установка:
```bash
pip install torch torchvision ultralytics transformers opencv-python scikit-learn pillow pandas
```

---

## 📈 Статистика

Для видео `25_12-20.mp4` (823 кадра):

| Параметр | Значение |
|----------|----------|
| Извлечено кадров | 164 (каждый 5-й) |
| Найдено ценников | 755 |
| Train/Val split | 131 / 33 |
| Время подготовки | ~5 мин |
| Время обучения (100 эпох) | ~30 мин |

Для папки с фото (50 изображений):

| Параметр | Значение |
|----------|----------|
| Обработано фото | 50 |
| Найдено ценников | ~200-300 |
| Train/Val split | 40 / 10 |
| Время подготовки | ~3 мин |
| Время обучения (100 эпох) | ~15 мин |

---

## 💡 Советы

1. **Начните с малого** — обработайте 1 видео/фото, обучите модель, проверьте качество
2. **Используйте `--color-filter`** — уменьшает ложные срабатывания
3. **Проверьте аннотации** — откройте несколько `.txt` файлов или detections.csv
4. **Early stopping** — модель сама остановится, если нет улучшений (patience=50)
5. **Аугментации** — YOLO автоматически применяет аугментации при обучении
6. **DETECT_EVERY** — увеличьте для ускорения обработки видео (каждый 10-20 кадр)
7. **ROTATE** — включите если видео снято вертикально
8. **prepare_dataset.py** — работает и с видео, и с изображениями

---

## 📞 Поддержка

При проблемах:
1. Проверьте наличие CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
2. Убедитесь, что видео/фото в папке имеют правильные расширения
3. Проверьте, что есть интернет для загрузки моделей
4. Очистите кэш: `pip cache purge`
