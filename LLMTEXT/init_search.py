"""
Инициализация гибридного поиска
Запускается 1 раз для создания эмбеддингов и индексов
"""

import sqlite3
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Конфигурация
DB_PATH = r"lenta_products.db"
MODEL_PATH = r"LLMTEXT/bge-m3"
FAISS_INDEX_PATH = r"LLMTEXT/faiss_index.bin"
PRODUCTS_CACHE_PATH = r"LLMTEXT/products_cache.pkl"
BM25_CACHE_PATH = r"LLMTEXT/bm25_cache.pkl"
MODEL_NAME = "deepvk/USER-bge-m3"

print("=" * 60)
print("ИНИЦИАЛИЗАЦИЯ ГИБРИДНОГО ПОИСКА")
print("=" * 60)

# Шаг 1: Загрузка модели
print("\n[1/6] Загрузка модели...")
model = SentenceTransformer(MODEL_NAME)
model.save(MODEL_PATH)
print(f"    Модель сохранена в {MODEL_PATH}")

# Шаг 2: Загрузка товаров из БД
print("\n[2/6] Загрузка товаров из БД...")
conn = sqlite3.connect(DB_PATH)
query = """
    SELECT 
        id, product_id, name, name_original, barcode, category, brand, price
    FROM products
    WHERE name IS NOT NULL AND name_original IS NOT NULL
"""
products_df = pd.read_sql_query(query, conn)
conn.close()
print(f"    Загружено {len(products_df)} товаров")

# Шаг 3: Создание текстов для эмбеддингов
print("\n[3/6] Подготовка текстов...")
product_texts = []
for idx, row in products_df.iterrows():
    text = f"{row['name']} {row['name_original']}"
    if pd.notna(row.get('brand')):
        text += f" {row['brand']}"
    product_texts.append(text)

print(f"    Создано {len(product_texts)} текстов")

# Шаг 4: Создание эмбеддингов
print("\n[4/6] Создание эмбеддингов (может занять 5-10 минут)...")
embeddings = model.encode(
    product_texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)
print(f"    Эмбеддинги созданы: {embeddings.shape}")

# Шаг 5: Построение FAISS индекса
print("\n[5/6] Построение FAISS HNSW индекса...")
dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(embeddings)
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"    FAISS индекс сохранён: {FAISS_INDEX_PATH}")

# Шаг 6: Сохранение кэша
print("\n[6/6] Сохранение кэша товаров...")
cache_data = {
    'products_df': products_df,
    'product_texts': product_texts,
    'created_at': datetime.now().isoformat()
}
with open(PRODUCTS_CACHE_PATH, 'wb') as f:
    pickle.dump(cache_data, f)
print(f"    Кэш сохранён: {PRODUCTS_CACHE_PATH}")

# Сохранение эмбеддингов в БД
print("\n[7/7] Сохранение эмбеддингов в БД...")
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_embeddings (
        product_id INTEGER PRIMARY KEY,
        embedding BLOB NOT NULL,
        text_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('DELETE FROM product_embeddings')

for idx, row in products_df.iterrows():
    embedding_bytes = embeddings[idx].tobytes()
    cursor.execute(
        'INSERT OR REPLACE INTO product_embeddings (product_id, embedding, text_hash) VALUES (?, ?, ?)',
        (row['product_id'], embedding_bytes, hash(product_texts[idx]))
    )

conn.commit()
conn.close()
print("    Эмбеддинги сохранены в БД")

# Создание таблицы OCR кэша
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS ocr_search_cache (
        ocr_text_hash TEXT PRIMARY KEY,
        ocr_text TEXT NOT NULL,
        result_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()
print("    Таблица OCR кэша создана")

print("\n" + "=" * 60)
print("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
print("=" * 60)
print(f"""
Файлы созданы:
- {MODEL_PATH}/ - модель
- {FAISS_INDEX_PATH} - FAISS индекс
- {PRODUCTS_CACHE_PATH} - кэш товаров
- {BM25_CACHE_PATH} - BM25 индекс (создаётся в ноутбуке)

Теперь откройте levenshein.ipynb и запустите ячейки!
""")
