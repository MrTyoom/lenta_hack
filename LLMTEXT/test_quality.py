import sys
sys.path.insert(0, 'LLMTEXT')
from product_matcher import find_top5_matches
import pandas as pd

# Тест 1: АГДАМ Портвейн белое vs красное
df1 = pd.DataFrame({'ocr_text': ['Вино креплёное ликерное АГДАМ Портвейн Резерви выдерж. бел (Азербайджан) 0,75Л']})
r1 = find_top5_matches(df1)
print('=== ТЕСТ 1: АГДАМ Портвейн белое ===')
for i in range(1, 6):
    name = r1[f'top{i}'].values[0]
    score = r1[f'top{i}_score'].values[0]
    print(f'{i}. {name} (score: {score})')

# Тест 2: ТИО ТОТО транслитерация
df2 = pd.DataFrame({'ocr_text': ['Винно крепленное ликерное ТИО ТОТО Опорос Херес обыкновенное (Испания) 0,75л']})
r2 = find_top5_matches(df2)
print()
print('=== ТЕСТ 2: ТИО ТОТО (транслит) ===')
for i in range(1, 6):
    name = r2[f'top{i}'].values[0]
    score = r2[f'top{i}_score'].values[0]
    print(f'{i}. {name} (score: {score})')

# Тест 3: Простой тест
df3 = pd.DataFrame({'ocr_text': ['Молоко домик в деревне 3.2%']})
r3 = find_top5_matches(df3)
print()
print('=== ТЕСТ 3: Молоко Домик в деревне ===')
for i in range(1, 6):
    name = r3[f'top{i}'].values[0]
    score = r3[f'top{i}_score'].values[0]
    print(f'{i}. {name} (score: {score})')
