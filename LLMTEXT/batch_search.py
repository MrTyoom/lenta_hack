"""
Скрипт для потоковой обработки батчей текстов
Использует гибридный поиск с нормализованным кэшем

Пример запуска:
    python batch_search.py --input ocr_texts.jsonl --output results.jsonl --top-k 5

Формат входного файла (JSONL, одна запись на строку):
    {"id": "001", "ocr_text": "Мед БЕРЕСТОВ 500г натуральный"}
    {"id": "002", "ocr_text": "Вино SAN VALENTIN Гарнача кр. сух. Испания"}

Формат выходного файла (JSON):
    {
      "processed_at": "2026-05-17T14:30:00",
      "total_processed": 100,
      "cache_hits": 45,
      "results": [...]
    }
"""

import argparse
import json
import sqlite3
import pickle
import base64
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from hybrid_search import HybridSearch, TextNormalizer


# === КОНФИГУРАЦИЯ ===
SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = str(SCRIPT_DIR / "lenta_products.db")
MODEL_PATH = str(SCRIPT_DIR / "bge-m3")
FAISS_INDEX_PATH = str(SCRIPT_DIR / "faiss_index.bin")
PRODUCTS_CACHE_PATH = str(SCRIPT_DIR / "products_cache.pkl")
BM25_CACHE_PATH = str(SCRIPT_DIR / "bm25_cache.pkl")


class BatchSearchEngine:
    """Обёртка над HybridSearch для пакетной обработки"""
    
    def __init__(self):
        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ ПОИСКОВОГО ДВИЖКА")
        print("=" * 60)
        
        self.search_engine = HybridSearch()
        self.stats = {
            'total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'barcodes': 0
        }
        
    def _check_cache_status(self, ocr_text: str) -> bool:
        """Проверяет, есть ли запрос в кэше (для статистики)"""
        normalized = self.search_engine.normalizer.normalize(ocr_text)
        text_hash = hashlib.md5(normalized.encode()).hexdigest()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT created_at FROM ocr_search_cache 
            WHERE ocr_text_hash = ?
        ''', (text_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            cache_time = datetime.fromisoformat(result[0])
            if datetime.now() - cache_time < self.search_engine.cache_ttl:
                return True
        
        return False
    
    def process_single(self, ocr_text: str, top_k: int = 5, use_cache: bool = True) -> Dict[str, Any]:
        """
        Обработка одного текста
        
        Returns:
            Dict с результатами поиска
        """
        self.stats['total'] += 1
        
        # Проверка кэша перед поиском (для статистики)
        if use_cache and self._check_cache_status(ocr_text):
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        
        try:
            results = self.search_engine.search(ocr_text, top_k=top_k, use_cache=use_cache)
            
            # Проверка, найден ли штрихкод
            if results and results[0].get('match_type') == 'barcode':
                self.stats['barcodes'] += 1
            
            return {
                'success': True,
                'ocr_text': ocr_text,
                'found_count': len(results),
                'results': results,
                'top_match': results[0] if results else None
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'success': False,
                'ocr_text': ocr_text,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def process_batch(self, texts: List[str], top_k: int = 5, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Обработка батча текстов
        
        Args:
            texts: Список текстов для обработки
            top_k: Сколько товаров возвращать для каждого запроса
            use_cache: Использовать ли кэширование
        
        Returns:
            Список результатов
        """
        results = []
        for i, text in enumerate(texts, 1):
            result = self.process_single(text, top_k, use_cache)
            results.append(result)
            
            # Прогресс
            if i % 100 == 0:
                print(f"  Обработано {i}/{len(texts)} ({i/len(texts)*100:.1f}%)")
        
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику обработки"""
        return self.stats.copy()


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Загрузка JSONL файла"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Oshibka v stroke {line_num}: {e}")
    return data


def process_file(input_path: str, output_path: str, top_k: int = 5, use_cache: bool = True):
    """
    Обработка входного JSONL файла
    
    Args:
        input_path: Путь к входному файлу (JSONL)
        output_path: Путь к выходному файлу (JSON)
        top_k: Количество товаров в результатах
        use_cache: Использовать ли кэширование
    """
    print("=" * 60)
    print("ПАКЕТНАЯ ОБРАБОТКА ТЕКСТОВ")
    print("=" * 60)
    
    # Загрузка данных
    print(f"\n[1/4] Загрузка данных из {input_path}...")
    input_data = load_jsonl(input_path)
    print(f"    Загружено записей: {len(input_data)}")
    
    if not input_data:
        print("[ERROR] Входной файл пуст или не содержит валидных записей")
        return
    
    # Инициализация движка
    print(f"\n[2/4] Инициализация поискового движка...")
    engine = BatchSearchEngine()
    
    # Обработка
    print(f"\n[3/4] Начало обработки...")
    start_time = time.time()
    
    results = []
    for i, item in enumerate(input_data, 1):
        ocr_text = item.get('ocr_text', '')
        request_id = item.get('id', str(i))
        
        if not ocr_text:
            print(f"[WARN] Propuscheno zapisi {request_id}: net ocr_text")
            continue
        
        result = engine.process_single(ocr_text, top_k=top_k, use_cache=use_cache)
        result['request_id'] = request_id
        
        results.append(result)
        
        # Progress
        if i % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            print(f"  Obrabotano {i}/{len(input_data)} ({i/len(input_data)*100:.1f}%) | "
                  f"Srednee vremya: {avg_time:.3f} sek")
    
    total_time = time.time() - start_time
    
    # Сохранение результатов
    print(f"\n[4/4] Сохранение результатов в {output_path}...")
    
    output = {
        'processed_at': datetime.now().isoformat(),
        'total_processed': len(results),
        'processing_time_sec': round(total_time, 3),
        'avg_time_per_query_sec': round(total_time / len(results), 3) if results else 0,
        'cache_hit_rate': round(engine.stats['cache_hits'] / engine.stats['total'] * 100, 1) if engine.stats['total'] > 0 else 0,
        'statistics': engine.get_stats(),
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    # Отчёт
    print("\n" + "=" * 60)
    print("OTCHET OB OBRABOTKE")
    print("=" * 60)
    print(f"   Vsego obrabotano: {engine.stats['total']}")
    print(f"   Cache hits: {engine.stats['cache_hits']} ({output['cache_hit_rate']}%)")
    print(f"   Cache misses: {engine.stats['cache_misses']}")
    print(f"   Naideno po shtrihkodu: {engine.stats['barcodes']}")
    print(f"   Oshibki: {engine.stats['errors']}")
    print(f"   Obwee vremya: {total_time:.3f} sek")
    print(f"   Srednee vremya na zapros: {total_time / len(results):.3f} sek")
    print(f"\n[OK] Rezultati sohraneni: {output_path}")
    print("=" * 60)


def process_stdin(top_k: int = 5, use_cache: bool = True):
    """
    Обработка текстов из stdin (для потокового режима)
    
    Использование:
        echo '{"ocr_text": "Мед БЕРЕСТОВ 500г"}' | python batch_search.py --stdin
    """
    engine = BatchSearchEngine()
    
    for line_num, line in enumerate(iter(input, ''), 1):
        try:
            data = json.loads(line.strip())
            ocr_text = data.get('ocr_text', '')
            request_id = data.get('id', str(line_num))
            
            if not ocr_text:
                print(json.dumps({'error': 'Нет ocr_text', 'request_id': request_id}, ensure_ascii=False))
                continue
            
            result = engine.process_single(ocr_text, top_k=top_k, use_cache=use_cache)
            result['request_id'] = request_id
            
            print(json.dumps(result, ensure_ascii=False))
            
        except json.JSONDecodeError as e:
            print(json.dumps({'error': f'Invalid JSON: {e}', 'line': line_num}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        description='Пакетная обработка текстов для поиска товаров',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обработка файла:
  python batch_search.py --input ocr_texts.jsonl --output results.jsonl

  # С другим количеством результатов:
  python batch_search.py -i input.jsonl -o output.jsonl --top-k 10

  # Без кэширования:
  python batch_search.py -i input.jsonl -o output.jsonl --no-cache

  # Потоковый режим (из stdin):
  echo '{"ocr_text": "Мед БЕРЕСТОВ 500г"}' | python batch_search.py --stdin
        """
    )
    
    parser.add_argument('-i', '--input', type=str, help='Входной файл (JSONL)')
    parser.add_argument('-o', '--output', type=str, help='Выходной файл (JSON)')
    parser.add_argument('--top-k', type=int, default=5, help='Количество товаров в результатах (по умолчанию: 5)')
    parser.add_argument('--no-cache', action='store_true', help='Отключить кэширование')
    parser.add_argument('--stdin', action='store_true', help='Потоковый режим (читать из stdin)')
    
    args = parser.parse_args()
    
    if args.stdin:
        process_stdin(top_k=args.top_k, use_cache=not args.no_cache)
    elif args.input and args.output:
        process_file(args.input, args.output, top_k=args.top_k, use_cache=not args.no_cache)
    else:
        parser.print_help()
        print("\n❌ Ошибка: Укажите --input и --output или используйте --stdin")


if __name__ == "__main__":
    main()
