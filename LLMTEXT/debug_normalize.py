import sys
sys.path.insert(0, '.')
from product_matcher import normalize_text, TRANSLIT_MAP

tests = [
    'Вино креплёное ликерное АГДАМ Портвейн Резерви выдерж. бел (Азербайджан) 0,75Л',
    'Винно крепленное ликерное ТИО ТОТО Опорос Херес обыкновенное (Испания) 0,75л',
    'Молоко домик в деревне 3.2%',
]

for t in tests:
    print(f'IN:  {t}')
    print(f'OUT: {normalize_text(t)}')
    print()
