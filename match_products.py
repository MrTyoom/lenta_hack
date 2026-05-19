"""
Скрипт для матчинга товаров из site_total.xlsx с db_hack.csv
Использует FAISS + bge-m3 модель + Левенштейн

Запуск:
    python match_products.py
"""

import os
import re
import time
import pickle
import logging
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer

# === КОНФИГУРАЦИЯ ===
SCRIPT_DIR = Path(__file__).parent.absolute()
LLMTEXT_DIR = SCRIPT_DIR / "LLMTEXT"

SITE_TOTAL_PATH = SCRIPT_DIR / "LLMTEXT" / "site_total.xlsx"
DB_HACK_PATH = SCRIPT_DIR / "LLMTEXT" / "db_hack.csv"
OUTPUT_PATH = SCRIPT_DIR / "matching_result.xlsx"

MODEL_PATH = str(LLMTEXT_DIR / "bge-m3")
FAISS_INDEX_PATH = str(LLMTEXT_DIR / "faiss_index_db_hack.bin")
DB_HACK_CACHE_PATH = str(LLMTEXT_DIR / "db_hack_cache.pkl")

MODEL_NAME = "deepvk/USER-bge-m3"

SIMILARITY_THRESHOLD = 0.8
TOP_K = 1

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "match_products.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Стоп-слова и нормализация
STOP_WORDS = {
    'шт', 'т', 'г', 'кг', 'л', 'мл', 'руб', 'цена', 'акция', 'скидка',
    'россия', 'испания', 'германия', 'китай', 'italy', 'france',
    'сух', 'кр', 'бел', 'сухое', 'красное', 'белое', 'полусладкое',
    'new', 'hit', 'top', 'sale', 'выгода'
}

TRANSLIT_MAP = str.maketrans({
    'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M',
    'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X',
    'а': 'a', 'в': 'b', 'е': 'e', 'к': 'k', 'м': 'm',
    'н': 'h', 'о': 'o', 'р': 'p', 'с': 'c', 'т': 't', 'х': 'x',
})


def normalize_text(text: str) -> str:
    """Нормализация текста для поиска"""
    if not text:
        return ""
    
    # Удаление цен и процентов
    text = re.sub(r'\b\d{3,5}\b', '', text)
    text = re.sub(r'\d+[,.]?\d*\s*%', '', text)
    text = re.sub(r'-\d+%', '', text)
    
    # Транслитерация
    text = text.translate(TRANSLIT_MAP)
    
    # Удаление спецсимволов
    text = re.sub(r'[^\w\sа-яА-Яa-zA-Z-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Lowercase и удаление стоп-слов
    words = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1]
    
    return ' '.join(words)


def load_or_create_model():
    """Загрузка или создание модели"""
    logger.info("Загрузка модели...")
    
    if os.path.exists(MODEL_PATH):
        logger.info(f"Модель найдена в {MODEL_PATH}")
        model = SentenceTransformer(MODEL_PATH)
    else:
        logger.info(f"Модель не найдена, загрузка {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        logger.info(f"Модель сохранена в {MODEL_PATH}")
    
    return model


def load_or_create_db_hack_index(model):
    """Загрузка или создание индекса для db_hack.csv"""
    
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DB_HACK_CACHE_PATH):
        logger.info("Индекс для db_hack.csv найден, загрузка...")
        
        with open(DB_HACK_CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
            db_hack_df = cache_data['db_hack_df']
            db_hack_texts = cache_data['db_hack_texts']
        
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        
        logger.info(f"Загружено {len(db_hack_df)} товаров из кэша")
        return db_hack_df, db_hack_texts, faiss_index
    
    # Создание нового индекса
    logger.info("Индекс не найден, создание нового...")
    logger.info(f"Загрузка {DB_HACK_PATH}...")
    
    db_hack_df = pd.read_csv(DB_HACK_PATH, sep=';', encoding='cp1251', dtype={'code': str})
    logger.info(f"Загружено {len(db_hack_df)} товаров")
    
    logger.info("Нормализация текстов...")
    db_hack_texts = [normalize_text(str(name)) for name in db_hack_df['fullname']]
    
    logger.info("Создание эмбеддингов...")
    embeddings = model.encode(
        db_hack_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    logger.info(f"Эмбеддинги созданы: {embeddings.shape}")
    
    logger.info("Построение FAISS индекса...")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexHNSWFlat(dimension, 16, faiss.METRIC_INNER_PRODUCT)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    logger.info(f"FAISS индекс сохранён: {FAISS_INDEX_PATH}")
    
    logger.info("Сохранение кэша...")
    cache_data = {
        'db_hack_df': db_hack_df,
        'db_hack_texts': db_hack_texts
    }
    with open(DB_HACK_CACHE_PATH, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"Кэш сохранён: {DB_HACK_CACHE_PATH}")
    
    return db_hack_df, db_hack_texts, faiss_index


def find_best_match(query_text: str, model, db_hack_df, db_hack_texts, faiss_index):
    """Поиск лучшего совпадения"""
    
    query_normalized = normalize_text(query_text)
    
    if not query_normalized:
        return None, 0.0
    
    # Эмбеддинг запроса
    query_embedding = model.encode([query_normalized], normalize_embeddings=True)
    
    # FAISS поиск
    D, I = faiss_index.search(query_embedding, TOP_K)
    
    if len(I[0]) == 0 or I[0][0] == -1:
        return None, 0.0
    
    best_idx = I[0][0]
    faiss_score = D[0][0]
    
    # Левенштейн для точности
    import Levenshtein
    db_text = db_hack_texts[best_idx]
    levenshtein_score = Levenshtein.ratio(query_normalized, db_text)
    
    # Комбинированный скор
    combined_score = 0.5 * faiss_score + 0.5 * levenshtein_score
    
    return best_idx, combined_score


def main():
    logger.info("=" * 60)
    logger.info("ЗАПУСК СКРИПТА МАТЧИНГА ТОВАРОВ")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Проверка файлов
    if not SITE_TOTAL_PATH.exists():
        logger.error(f"Файл не найден: {SITE_TOTAL_PATH}")
        return
    
    if not DB_HACK_PATH.exists():
        logger.error(f"Файл не найден: {DB_HACK_PATH}")
        return
    
    # Загрузка модели
    model = load_or_create_model()
    
    # Загрузка/создание индекса
    db_hack_df, db_hack_texts, faiss_index = load_or_create_db_hack_index(model)
    
    # Загрузка site_total.xlsx
    logger.info(f"Загрузка {SITE_TOTAL_PATH}...")
    site_total_df = pd.read_excel(SITE_TOTAL_PATH)
    logger.info(f"Загружено {len(site_total_df)} товаров для матчинга")
    
    # Проверка колонок
    if 'name' not in site_total_df.columns:
        logger.error("Колонка 'name' не найдена в site_total.xlsx")
        return
    
    if 'sku' not in site_total_df.columns:
        logger.error("Колонка 'sku' не найдена в site_total.xlsx")
        return
    
    # Матчинг
    logger.info("Начало матчинга...")
    
    results = []
    matched_count = 0
    not_matched_count = 0
    
    total_rows = len(site_total_df)
    
    for idx, row in site_total_df.iterrows():
        name = str(row['name'])
        sku = row['sku']
        
        best_idx, score = find_best_match(
            name, model, db_hack_df, db_hack_texts, faiss_index
        )
        
        if best_idx is not None and score >= SIMILARITY_THRESHOLD:
            code = db_hack_df.iloc[best_idx]['code']
            matched_count += 1
        else:
            code = ""
            not_matched_count += 1
        
        results.append({
            'name': name,
            'sku': sku,
            'code': code,
            'score': round(score, 4) if best_idx is not None else 0.0
        })
        
        # Прогресс
        if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            progress = (idx + 1) / total_rows * 100
            logger.info(
                f"Прогресс: {idx + 1}/{total_rows} ({progress:.1f}%) | "
                f"Найдено: {matched_count} | Не найдено: {not_matched_count} | "
                f"Среднее время: {avg_time:.3f} сек"
            )
    
    # Создание результирующего DataFrame
    result_df = pd.DataFrame(results)
    
    # Сохранение
    logger.info(f"Сохранение результата в {OUTPUT_PATH}...")
    result_df.to_excel(OUTPUT_PATH, index=False)
    
    # Итоговая статистика
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info("МАТЧИНГ ЗАВЕРШЕН")
    logger.info("=" * 60)
    logger.info(f"Всего товаров: {total_rows}")
    logger.info(f"Найдено совпадений (score >= {SIMILARITY_THRESHOLD}): {matched_count}")
    logger.info(f"Не найдено совпадений: {not_matched_count}")
    logger.info(f"Процент совпадений: {matched_count / total_rows * 100:.1f}%")
    logger.info(f"Общее время: {total_time:.2f} сек")
    logger.info(f"Среднее время на товар: {total_time / total_rows:.4f} сек")
    logger.info(f"Результат сохранён: {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
