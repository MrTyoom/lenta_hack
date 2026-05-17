"""
ГИБРИДНЫЙ ПОИСК ТОВАРОВ ПО OCR ТЕКСТУ
FAISS + Левенштейн + BM25

Использование:
    search_engine = HybridSearch()
    results = search_engine.search("Мед БЕРЕСТОВ 500г", top_k=5)
"""

import sqlite3
import pickle
import numpy as np
import pandas as pd
import faiss
import Levenshtein
import re
import base64
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import time

# Конфигурация
DB_PATH = r"lenta_products.db"
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_PATH = str(SCRIPT_DIR / "bge-m3")
FAISS_INDEX_PATH = str(SCRIPT_DIR / "faiss_index.bin")
PRODUCTS_CACHE_PATH = str(SCRIPT_DIR / "products_cache.pkl")
BM25_CACHE_PATH = str(SCRIPT_DIR / "bm25_cache.pkl")

TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5
CACHE_TTL_HOURS = 24


class TextNormalizer:
    """Нормализация текста для OCR"""
    
    def __init__(self):
        self.stop_words = {
            'шт', 'т', 'г', 'кг', 'л', 'мл', 'руб', '₽', 'цена', 'акция',
            'скидка', 'выгода', 'new', 'hit', 'top', 'sale',
            'россия', 'испания', 'германия', 'китай', 'italy', 'france',
            'сух', 'кр', 'бел', 'сухое', 'красное', 'белое'
        }
    
    def remove_prices(self, text):
        text = re.sub(r'\b\d{3,5}\b', '', text)
        text = re.sub(r'-\d+%', '', text)
        text = re.sub(r'\d+\.\d+', '', text)
        return text
    
    def remove_special_chars(self, text):
        text = re.sub(r'[^\w\sа-яА-Яa-zA-ZёЁ-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def remove_stopwords(self, text):
        words = text.lower().split()
        filtered = [w for w in words if w not in self.stop_words and len(w) > 1]
        return ' '.join(filtered)
    
    def normalize(self, text, do_stopwords=True):
        text = self.remove_prices(text)
        text = self.remove_special_chars(text)
        text = text.lower()
        if do_stopwords:
            text = self.remove_stopwords(text)
        return text


class HybridSearch:
    """Гибридный поиск: FAISS + Левенштейн + BM25"""
    
    def __init__(self):
        print("Инициализация гибридного поиска...")
        
        # Загрузка модели
        self.model = SentenceTransformer(MODEL_PATH)
        
        # Загрузка FAISS
        self.faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Загрузка кэша товаров
        with open(PRODUCTS_CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
            self.products_df = cache_data['products_df']
            self.product_texts = cache_data['product_texts']
        
        # Загрузка BM25
        with open(BM25_CACHE_PATH, 'rb') as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data['bm25']
        
        # Нормализатор
        self.normalizer = TextNormalizer()
        
        # Кэш OCR
        self.cache_ttl = timedelta(hours=CACHE_TTL_HOURS)
        self._init_ocr_cache_table()
        
        print("OK Гибридный поиск готов")
    
    def _init_ocr_cache_table(self):
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
    
    def _get_from_cache(self, ocr_text):
        normalized = self.normalizer.normalize(ocr_text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT result_json, created_at 
            FROM ocr_search_cache 
            WHERE ocr_text_hash = ?
        ''', (text_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            result_b64, created_at = result
            cache_time = datetime.fromisoformat(created_at)
            if datetime.now() - cache_time < self.cache_ttl:
                print("OK Найдено в кэше")
                return pickle.loads(base64.b64decode(result_b64))
            else:
                print("Кэш устарел")
                self._clear_cache(text_hash)
        
        return None
    
    def _save_to_cache(self, ocr_text, result):
        normalized = self.normalizer.normalize(ocr_text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        result_b64 = base64.b64encode(pickle.dumps(result)).decode('ascii')
        cursor.execute('''
            INSERT OR REPLACE INTO ocr_search_cache 
            (ocr_text_hash, ocr_text, result_json, created_at)
            VALUES (?, ?, ?, ?)
        ''', (text_hash, ocr_text, result_b64, datetime.now().isoformat()))
        conn.commit()
        conn.close()
    
    def _clear_cache(self, text_hash=None):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        if text_hash:
            cursor.execute('DELETE FROM ocr_search_cache WHERE ocr_text_hash = ?', (text_hash,))
        else:
            cursor.execute('DELETE FROM ocr_search_cache')
        conn.commit()
        conn.close()
    
    def _search_by_barcode(self, barcode):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, product_id, name, name_original, brand, barcode, category, price
            FROM products
            WHERE barcode = ? OR barcode LIKE ?
        ''', (barcode, f'%{barcode}%'))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            print(f"OK Найдено по штрихкоду: {len(results)}")
            return [{
                'id': r[0], 'product_id': r[1], 'name': r[2], 'name_original': r[3],
                'brand': r[4], 'barcode': r[5], 'category': r[6], 'price': r[7],
                'match_type': 'barcode', 'confidence': 1.0
            } for r in results]
        
        return None
    
    def _extract_barcode_from_ocr(self, ocr_text):
        ean13_pattern = r'\b(\d{13})\b'
        ean8_pattern = r'\b(\d{8})\b'
        
        matches = re.findall(ean13_pattern, ocr_text)
        if matches:
            return matches[0]
        
        matches = re.findall(ean8_pattern, ocr_text)
        if matches:
            return matches[0]
        
        return None
    
    def search(self, ocr_text, top_k=TOP_K_FINAL, use_cache=True):
        """
        Гибридный поиск товаров по OCR тексту
        """
        start_time = time.time()
        
        # 1. Проверка кэша
        if use_cache:
            cached_result = self._get_from_cache(ocr_text)
            if cached_result:
                return cached_result
        
        # 2. Поиск по штрихкоду
        barcode = self._extract_barcode_from_ocr(ocr_text)
        if barcode:
            print(f"Найден штрихкод: {barcode}")
            barcode_results = self._search_by_barcode(barcode)
            if barcode_results:
                return barcode_results[:top_k]
        
        # 3. Нормализация текста
        query_normalized = self.normalizer.normalize(ocr_text)
        
        # 4. Эмбеддинг запроса
        query_embedding = self.model.encode([query_normalized], normalize_embeddings=True)
        
        # 5. FAISS поиск
        D, I = self.faiss_index.search(query_embedding, TOP_K_CANDIDATES)
        candidates = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.products_df):
                row = self.products_df.iloc[idx]
                candidates.append({
                    'idx': idx,
                    'product_id': row['product_id'],
                    'name': row['name'],
                    'name_original': row['name_original'],
                    'brand': row.get('brand'),
                    'category': row.get('category'),
                    'price': row.get('price'),
                    'faiss_score': float(score)
                })
        
        # 6. BM25 фильтрация
        query_tokens = query_normalized.split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        for candidate in candidates:
            candidate['bm25_score'] = float(bm25_scores[candidate['idx']])
        
        bm25_threshold = np.percentile(bm25_scores, 70)
        candidates = [c for c in candidates if c['bm25_score'] > bm25_threshold]
        
        # 7. Левенштейн ранжирование
        for candidate in candidates:
            dist_name = Levenshtein.ratio(query_normalized, str(candidate['name']).lower())
            dist_name_orig = Levenshtein.ratio(query_normalized, str(candidate['name_original']).lower())
            candidate['levenshtein_score'] = max(dist_name, dist_name_orig)
            
            # Комбинированный скор
            candidate['combined_score'] = (
                0.4 * candidate['faiss_score'] +
                0.3 * candidate['bm25_score'] +
                0.3 * candidate['levenshtein_score']
            )
        
        # Сортировка
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 8. Формирование результата
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            results.append({
                'rank': i + 1,
                'product_id': candidate['product_id'],
                'name': candidate['name'],
                'name_original': candidate['name_original'],
                'brand': candidate['brand'],
                'category': candidate['category'],
                'price': candidate['price'],
                'confidence': float(candidate['combined_score']),
                'faiss_score': candidate['faiss_score'],
                'bm25_score': candidate['bm25_score'],
                'levenshtein_ratio': candidate['levenshtein_score'],
                'match_type': 'hybrid'
            })
        
        # 9. Сохранение в кэш
        if use_cache and results:
            self._save_to_cache(ocr_text, results)
        
        elapsed = time.time() - start_time
        print(f"Время поиска: {elapsed:.3f} сек")
        print(f"Найдено товаров: {len(results)}")
        
        return results


# Тестирование
if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТ ГИБРИДНОГО ПОИСКА")
    print("=" * 60)
    
    search_engine = HybridSearch()
    
    # Тест 1: Вино
    print("\n" + "=" * 60)
    print("ТЕСТ 1: Поиск вина")
    print("=" * 60)
    
    test1 = "Вино SAN VALENTIN Гарнача кр. сух. (Испания) 0.75L -25% 1223"
    results = search_engine.search(test1, top_k=5)
    
    for r in results:
        print(f"\n{r['rank']}. {r['name'][:60]}")
        print(f"   Бренд: {r['brand']}")
        print(f"   Цена: {r['price']}")
        print(f"   Confidence: {r['confidence']:.3f}")
        print(f"   Левенштейн: {r['levenshtein_ratio']:.3f}")
    
    # Тест 2: Мёд
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Поиск мёда")
    print("=" * 60)
    
    test2 = "Мед БЕРЕСТОВ А.С. 500г натуральный"
    results = search_engine.search(test2, top_k=5)
    
    for r in results:
        print(f"\n{r['rank']}. {r['name'][:60]}")
        print(f"   Бренд: {r['brand']}")
        print(f"   Цена: {r['price']}")
        print(f"   Confidence: {r['confidence']:.3f}")
    
    # Тест 3: Конфитюр
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Поиск конфитюра")
    print("=" * 60)
    
    test3 = "Конфитюр ZUEGG Клубника экстра 320г"
    results = search_engine.search(test3, top_k=5)
    
    for r in results:
        print(f"\n{r['rank']}. {r['name'][:60]}")
        print(f"   Бренд: {r['brand']}")
        print(f"   Цена: {r['price']}")
        print(f"   Confidence: {r['confidence']:.3f}")
    
    print("\n" + "=" * 60)
    print("ТЕСТЫ ЗАВЕРШЕНЫ")
    print("=" * 60)
