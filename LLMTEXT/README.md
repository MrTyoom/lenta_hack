# Поиск товаров Lenta по OCR тексту

Гибридный поиск товаров сети "Лента" по тексту с ценников (OCR).
Комбинирует FAISS (семантический поиск) + BM25 (ключевые слова) + Levenshtein (нечёткое сравнение).

## Структура проекта

```
LLMTEXT/
├── hybrid_search.py       # Основной движок гибридного поиска
├── batch_search.py        # Пакетная обработка батчей текстов
├── product_matcher.py     # Простой матчер для pandas DataFrame (top-5)
├── init_search.py         # Инициализация: эмбеддинги, FAISS, BM25
├── create_bm25.py         # Создание BM25 индекса
├── test_search.py         # Тестовые примеры поиска
├── lenta_products.db      # SQLite база товаров
├── site_total.xlsx        # Исходные данные (Excel)
├── bge-m3/                # ML модель эмбеддингов
├── faiss_index.bin        # Векторный индекс FAISS
├── products_cache.pkl     # Кэш товаров
├── bm25_cache.pkl         # BM25 индекс
├── examples/              # Примеры входных данных
├── output/                # Результаты обработки
├── docs/                  # Документация
└── temp/                  # Временные/вспомогательные скрипты
```

## Быстрый старт

### 1. Поиск одного текста

```python
from hybrid_search import HybridSearch

engine = HybridSearch()
results = engine.search("Мед БЕРЕСТОВ 500г натуральный", top_k=5)
for r in results:
    print(f"{r['rank']}. {r['name']} (confidence: {r['confidence']:.3f})")
```

### 2. Пакетная обработка файла

```bash
python batch_search.py --input examples/sample_input.jsonl --output output/results.json
```

### 3. Поиск по pandas DataFrame

```python
import pandas as pd
from product_matcher import find_top5_matches

df = pd.DataFrame({'ocr_text': ['молоко домик в деревне', 'хлеб бородинский']})
result = find_top5_matches(df)
# Результат: исходный df + столбцы top1..top5 и top1_sku..top5_sku
```

## Параметры batch_search.py

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--input` / `-i` | Входной JSONL файл | Обязательный* |
| `--output` / `-o` | Выходной JSON файл | Обязательный* |
| `--top-k` | Кол-во товаров в результатах | `5` |
| `--no-cache` | Отключить кэширование | `false` |
| `--stdin` | Потоковый режим | `false` |

*не нужны при использовании `--stdin`

## Формат входных данных (JSONL)

```jsonl
{"id": "001", "ocr_text": "Мед БЕРЕСТОВ 500г натуральный"}
{"id": "002", "ocr_text": "Вино SAN VALENTIN Гарнача кр. сух. 0.75L"}
```

## Требования

```bash
pip install sentence-transformers faiss-cpu rank-bm25 Levenshtein pandas rapidfuzz openpyxl
```

## Переинициализация

```bash
python init_search.py    # Пересоздать эмбеддинги и FAISS индекс
python create_bm25.py    # Пересоздать BM25 индекс
```
