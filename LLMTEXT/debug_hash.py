import pickle
import sys
sys.path.insert(0, '.')
from product_matcher import normalize_text

with open('name_hash_table.pkl', 'rb') as f:
    hash_table = pickle.load(f)

print(f'Размер хэш-таблицы: {len(hash_table)}')

# Ищем АГДАМ
for key in list(hash_table.keys())[:20]:
    if 'agdam' in key.lower() or 'port' in key.lower():
        print(f'Ключ: {key} -> {len(hash_table[key])} продуктов')

# Ищем херес
for key in list(hash_table.keys())[:50]:
    if 'xeres' in key.lower() or 'heres' in key.lower() or 'tio' in key.lower():
        print(f'Ключ: {key} -> {len(hash_table[key])} продуктов')

# Проверка конкретного запроса
test_query = 'Вино креплёное ликерное АГДАМ Портвейн Резерви выдерж. бел (Азербайджан) 0,75Л'
normalized = normalize_text(test_query)
print(f'\nНормализованный запрос: {normalized}')
print(f'Есть в хэш-таблице: {normalized in hash_table}')
