# Обучение модели классификации отделов

## Установка

```bash
cd train_model
pip install -r requirements.txt
```

## Запуск обучения

```bash
python train.py
```

## Структура

```
train_model/
├── config.py          # Настройки
├── dataset.py         # Загрузка данных
├── model.py           # Модели
├── train.py           # Главный скрипт
├── requirements.txt   # Зависимости
└── models/            # Сохранённые модели
    ├── best_model.pth
    ├── last_model.pth
    ├── training_history.json
    └── class_names.json
```

## Настройки (config.py)

- `MODEL_NAME`: 'efficientnet-b0' (рекомендуется), 'efficientnet-b3', 'resnet-50'
- `BATCH_SIZE`: 32 (уменьши если нет памяти)
- `NUM_EPOCHS`: 15
- `IMG_SIZE`: 224

## Результат

После обучения в папке `models/` будут:
- `best_model.pth` - лучшая модель
- `class_names.json` - имена классов
- `training_history.json` - история обучения
