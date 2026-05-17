import sqlite3
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import Levenshtein
import re

SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = str(SCRIPT_DIR / 'lenta_products.db')
EXCEL_PATH = str(SCRIPT_DIR / 'site_total.xlsx')
MODEL_PATH = str(SCRIPT_DIR / 'bge-m3')
FAISS_INDEX_PATH = str(SCRIPT_DIR / 'faiss_index.bin')
PRODUCTS_CACHE_PATH = str(SCRIPT_DIR / 'products_cache.pkl')
BM25_CACHE_PATH = str(SCRIPT_DIR / 'bm25_cache.pkl')
MODEL_NAME = 'deepvk/USER-bge-m3'

stop_words = {
    'шт', 'т', 'г', 'кг', 'л', 'мл', 'руб', 'цена', 'акция', 'скидка',
    'россия', 'испания', 'германия', 'сух', 'кр', 'бел'
}


HASH_TABLE_PATH = str(SCRIPT_DIR / 'name_hash_table.pkl')


def normalize_text(t):
    t = re.sub(r'\b\d{3,5}\b', '', t)
    t = re.sub(r'\d+[,.]?\d*\s*%', '', t)
    t = re.sub(r'[^\w\sа-яА-Яa-zA-Z-]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    words = [w for w in t.lower().split() if w not in stop_words and len(w) > 1]
    return ' '.join(words)


def build_hash_table(product_texts):
    print('Хэширование названий...')
    hash_table = {}
    for idx, text in enumerate(product_texts):
        normalized = normalize_text(text)
        if normalized not in hash_table:
            hash_table[normalized] = []
        hash_table[normalized].append(idx)
    print(f'Хэш-таблица: {len(hash_table)} уникальных названий')
    return hash_table


def init_db_from_excel():
    if os.path.exists(DB_PATH):
        return

    print('Создание базы из Excel...')
    df = pd.read_excel(EXCEL_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('site_products', conn, index=False, if_exists='replace')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_name ON site_products(name)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_sku ON site_products(sku)')
    conn.commit()
    conn.close()
    print(f'База создана: {len(df)} товаров')


def init_search_indexes():
    if not os.path.exists(DB_PATH):
        for p in [FAISS_INDEX_PATH, PRODUCTS_CACHE_PATH, BM25_CACHE_PATH, HASH_TABLE_PATH]:
            if os.path.exists(p):
                os.remove(p)
                print(f'Удалён старый кэш: {os.path.basename(p)}')

    need_init = not (
        os.path.exists(FAISS_INDEX_PATH) and
        os.path.exists(PRODUCTS_CACHE_PATH) and
        os.path.exists(BM25_CACHE_PATH) and
        os.path.exists(HASH_TABLE_PATH)
    )

    if not need_init:
        return

    init_db_from_excel()

    print('Загрузка модели...')
    if os.path.exists(MODEL_PATH):
        model = SentenceTransformer(MODEL_PATH)
    else:
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)

    print('Загрузка товаров...')
    conn = sqlite3.connect(DB_PATH)
    products_df = pd.read_sql('SELECT name, sku FROM site_products', conn)
    conn.close()

    product_texts = products_df['name'].astype(str).tolist()
    print(f'Товаров: {len(product_texts)}')

    print('Создание эмбеддингов...')
    embeddings = model.encode(
        product_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print('Построение FAISS индекса...')
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print('Сохранение кэша товаров...')
    with open(PRODUCTS_CACHE_PATH, 'wb') as f:
        pickle.dump({'products_df': products_df, 'product_texts': product_texts}, f)

    print('Построение BM25 индекса...')
    tokenized = [normalize_text(t).split() for t in product_texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_CACHE_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'tokenized_texts': tokenized}, f)

    print('Создание хэш-таблицы названий...')
    hash_table = build_hash_table(product_texts)
    with open(HASH_TABLE_PATH, 'wb') as f:
        pickle.dump(hash_table, f)

    print('Индексы готовы')


def find_top5_matches(input_df: pd.DataFrame, ocr_col: str = 'ocr_text') -> pd.DataFrame:
    init_search_indexes()

    model = SentenceTransformer(MODEL_PATH)

    with open(FAISS_INDEX_PATH, 'rb') as f:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(PRODUCTS_CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
        products_df = cache['products_df']
        product_texts = cache['product_texts']

    with open(BM25_CACHE_PATH, 'rb') as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data['bm25']

    with open(HASH_TABLE_PATH, 'rb') as f:
        hash_table = pickle.load(f)

    results = []

    for _, row in input_df.iterrows():
        ocr_text = str(row[ocr_col])
        query = normalize_text(ocr_text)

        if query in hash_table:
            candidates = []
            for idx in hash_table[query]:
                name = products_df.iloc[idx]['name']
                lev = Levenshtein.ratio(query, str(name).lower())
                candidates.append({
                    'name': name,
                    'sku': products_df.iloc[idx]['sku'],
                    'score': round(0.5 + 0.5 * lev, 4)
                })
            candidates.sort(key=lambda x: x['score'], reverse=True)
            top = candidates[:5]
        else:
            query_emb = model.encode([query], normalize_embeddings=True)
            D, I = faiss_index.search(query_emb, 20)

            query_tokens = query.split()
            bm25_scores = bm25.get_scores(query_tokens)
            bm25_threshold = np.percentile(bm25_scores, 70)

            candidates = []
            for idx, score in zip(I[0], D[0]):
                if idx < len(products_df) and bm25_scores[idx] > bm25_threshold:
                    name = products_df.iloc[idx]['name']
                    lev_score = Levenshtein.ratio(query, str(name).lower())
                    combined = 0.4 * score + 0.3 * bm25_scores[idx] + 0.3 * lev_score
                    candidates.append({
                        'name': name,
                        'sku': products_df.iloc[idx]['sku'],
                        'score': combined
                    })

            candidates.sort(key=lambda x: x['score'], reverse=True)
            top = candidates[:5]

        match = {}
        for i in range(1, 6):
            if i <= len(top):
                match[f'top{i}'] = top[i - 1]['name']
                match[f'top{i}_sku'] = top[i - 1]['sku']
                match[f'top{i}_score'] = round(top[i - 1]['score'], 4)
            else:
                match[f'top{i}'] = None
                match[f'top{i}_sku'] = None
                match[f'top{i}_score'] = None

        results.append(match)

    match_df = pd.DataFrame(results)
    return pd.concat([input_df.reset_index(drop=True), match_df], axis=1)


if __name__ == '__main__':
    test_df = pd.DataFrame({'ocr_text': ['молоко домик в деревне 3 2', 'хлеб бородинский']})
    result = find_top5_matches(test_df)
    print(result)
