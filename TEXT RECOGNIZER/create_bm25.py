"""Создание BM25 индекса"""
import pickle
from rank_bm25 import BM25Okapi
import re

stop_words = {'шт', 'т', 'г', 'кг', 'л', 'мл', 'руб', 'цена', 'акция', 'скидка', 'россия', 'испания', 'германия'}

def normalize(t):
    t = re.sub(r'\b\d{3,5}\b', '', t)
    t = re.sub(r'[^\w\sа-яА-Яa-zA-Z-]', ' ', t)
    t = re.sub(r'\s+', ' ', t)
    words = [w for w in t.lower().split() if w not in stop_words and len(w) > 1]
    return words

with open('LLMTEXT/products_cache.pkl', 'rb') as f:
    d = pickle.load(f)

texts = [normalize(t) for t in d['product_texts']]
print(f"Токенизировано {len(texts)} текстов")

bm25 = BM25Okapi(texts)
print("BM25 индекс построен")

with open('LLMTEXT/bm25_cache.pkl', 'wb') as f:
    pickle.dump({'bm25': bm25, 'tokenized_texts': texts}, f)

print("BM25 сохранён в LLMTEXT/bm25_cache.pkl")
