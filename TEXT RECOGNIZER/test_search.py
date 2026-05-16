"""
Тест гибридного поиска по OCR тексту
Запустите после инициализации (init_search.py)
"""

import pickle
import faiss
from sentence_transformers import SentenceTransformer
import re

# Конфигурация
DB_PATH = r"lenta_products.db"
MODEL_PATH = r"LLMTEXT/bge-m3"
FAISS_INDEX_PATH = r"LLMTEXT/faiss_index.bin"
PRODUCTS_CACHE_PATH = r"LLMTEXT/products_cache.pkl"

print("=" * 60)
print("ЗАГРУЗКА ПОИСКОВОГО ДВИЖКА")
print("=" * 60)

# Загрузка модели
print("Загрузка модели...")
model = SentenceTransformer(MODEL_PATH)

# Загрузка FAISS
print("Загрузка FAISS индекса...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Загрузка кэша товаров
print("Загрузка кэша товаров...")
with open(PRODUCTS_CACHE_PATH, 'rb') as f:
    cache_data = pickle.load(f)
    products_df = cache_data['products_df']
    product_texts = cache_data['product_texts']

print(f"OK Zagruzheno {len(products_df)} tovarov")


class SimpleNormalizer:
    """Простая нормализация без лемматизации"""
    
    def __init__(self):
        self.stop_words = {
            'шт', 'т', 'г', 'кг', 'л', 'мл', 'руб', '₽', 'цена', 'акция',
            'скидка', 'выгода', 'new', 'hit', 'top', 'sale',
            'россия', 'испания', 'германия', 'китай', 'italy', 'france',
            'сух', 'кр', 'бел', 'сухое', 'красное', 'белое'
        }
    
    def normalize(self, text):
        # Удаляем цены
        text = re.sub(r'\b\d{3,5}\b', '', text)
        text = re.sub(r'-\d+%', '', text)
        text = re.sub(r'[^\w\sа-яА-Яa-zA-ZёЁ-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        # Удаляем стоп-слова
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        return ' '.join(words)


normalizer = SimpleNormalizer()


def search(ocr_text, top_k=5):
    """Поиск товаров по OCR тексту"""
    
    # Нормализация
    query = normalizer.normalize(ocr_text)
    print(f"\nНормализованный запрос: {query[:100]}...")
    
    # Эмбеддинг
    query_emb = model.encode([query], normalize_embeddings=True)
    
    # FAISS поиск
    D, I = faiss_index.search(query_emb, top_k)
    
    # Результаты
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(products_df):
            row = products_df.iloc[idx]
            results.append({
                'name': row['name'],
                'name_original': row['name_original'],
                'brand': row.get('brand'),
                'category': row.get('category'),
                'price': row.get('price'),
                'product_id': row.get('product_id'),
                'score': float(score)
            })
    
    return results



test1 = """
Конфктюр  ZUEGG  Ктуонха жстра  (Гор)  320  25Д  259  23
"""

results = search(test1, top_k=5)
for i, r in enumerate(results, 1):
    print(f"\n{i}. {r['name'][:70]}")
    print(f"   Бренд: {r['brand']}")
    print(f"   Цена: {r['price']}")
    print(f"   Артикул: {r['product_id']}")
    print(f"   Score: {r['score']:.3f}")
