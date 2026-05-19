"""
Скрипт для матчинга товаров между листами data и db_hack в match.xlsx
Использует FAISS + bge-m3 + Левенштейн, всегда берёт top-1
"""
import os
import re
import time
import logging
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent.absolute()
LLMTEXT_DIR = SCRIPT_DIR / "LLMTEXT"

INPUT_PATH = SCRIPT_DIR / "match.xlsx"
OUTPUT_PATH = SCRIPT_DIR / "match_result.xlsx"

MODEL_PATH = str(LLMTEXT_DIR / "bge-m3")
FAISS_INDEX_PATH = str(LLMTEXT_DIR / "faiss_index_match.bin")
DB_HACK_NPY_PATH = str(LLMTEXT_DIR / "db_hack_texts_match.npy")

MODEL_NAME = "deepvk/USER-bge-m3"

TOP_K = 5  # candidates from FAISS, then re-rank with Levenshtein

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "match_v2.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Полная транслитерация кириллицы в латиницу (поддерживает многобуквенные замены)
CYR_TO_LAT_MAP = {
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
    'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K',
    'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
    'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'Ts',
    'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
    'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
    'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
    'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
    'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
    'э': 'e', 'ю': 'yu', 'я': 'ya',
}

def _translit(text: str) -> str:
    """Побуквенная транслитерация с поддержкой многобуквенных замен"""
    result = []
    for ch in text:
        result.append(CYR_TO_LAT_MAP.get(ch, ch))
    return ''.join(result)

def normalize_text(text: str) -> str:
    """Нормализация: транслит, чистка от мусора, lowercase"""
    if not text:
        return ""

    # Полная транслитерация
    text = _translit(text)

    # Удаляем размеры/веса: 250г, 1кг, 500мл, 1л, 1000г
    text = re.sub(r'\b\d+\s*(г|кг|мл|л|шт|w|ml|l|g|kg)\b', '', text, flags=re.IGNORECASE)
    # Удаляем "х N" (количество в упаковке): 4х, 6х, 12х и т.д.
    text = re.sub(r'\b\d+\s*[xхXХ×]\b', '', text)
    # Удаляем размеры типа 4*125г
    text = re.sub(r'\b\d+\s*\*\s*\d+\s*(г|кг|мл|л)?\b', '', text)
    # Удаляем сокращения упаковки
    text = re.sub(r'\b(м/у|м\/у|ст/б|ст\/б|ж/б|ж\/б|п/э|п\/э|пэт|дой-пак|пауч|бан|кор|уп)\b', '', text, flags=re.IGNORECASE)
    # Удаляем "Россия", "Германия" и т.п. в скобках
    text = re.sub(r'\([^)]*\)', ' ', text)
    # Удаляем проценты и цены
    text = re.sub(r'\d+[,.]?\d*\s*%', '', text)
    text = re.sub(r'-\d+%', '', text)
    # Удаляем оставшиеся цифры (обычно это артикулы или неинформативные числа)
    text = re.sub(r'\b\d+\b', '', text)
    # Спецсимволы → пробел
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Схлопываем пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    # Удаляем слова из 1 буквы (остатки мусора)
    text = ' '.join(w for w in text.lower().split() if len(w) > 1)

    return text


def load_model():
    model_path = os.path.join(LLMTEXT_DIR, "bge-m3")
    if os.path.exists(model_path):
        logger.info(f"Загрузка модели из {model_path}")
        return SentenceTransformer(model_path)
    logger.info(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(model_path)
    logger.info(f"Модель сохранена в {model_path}")
    return model


def load_db_hack(db_path, sheet):
    df = pd.read_excel(db_path, sheet_name=sheet, dtype={'code': str})
    logger.info(f"Загружено {len(df)} товаров из db_hack")
    df['code'] = df['code'].astype(str).str.replace(r'\.0$', '', regex=True)
    return df


def build_index(db_df, model):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DB_HACK_NPY_PATH):
        logger.info("Загрузка кэшированного индекса...")
        texts = np.load(DB_HACK_NPY_PATH, allow_pickle=True).tolist()
        index = faiss.read_index(FAISS_INDEX_PATH)
        logger.info(f"Индекс загружен: {len(texts)} текстов")
        return texts, index

    logger.info("Создание индекса FAISS...")
    texts = [normalize_text(str(n)) for n in db_df['fullname']]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Эмбеддинги: {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(DB_HACK_NPY_PATH, np.array(texts, dtype=object))
    logger.info(f"Индекс сохранён: {FAISS_INDEX_PATH}")
    return texts, index


def match_one(query_text, model, db_df, db_texts, faiss_index):
    q = normalize_text(query_text)
    if not q:
        return -1, -1.0

    q_emb = model.encode([q], normalize_embeddings=True)
    D, I = faiss_index.search(q_emb, TOP_K)

    import Levenshtein
    best_idx, best_score = -1, -1.0
    for i in range(len(I[0])):
        idx = I[0][i]
        if idx == -1:
            continue
        faiss_s = float(D[0][i])
        lev_s = Levenshtein.ratio(q, db_texts[idx])
        combined = 0.4 * faiss_s + 0.6 * lev_s
        if combined > best_score:
            best_score = combined
            best_idx = idx

    return best_idx, best_score


def main():
    logger.info("=" * 60)
    logger.info("МАТЧИНГ ТОВАРОВ V2 (match.xlsx: data ↔ db_hack)")
    logger.info("=" * 60)

    if not INPUT_PATH.exists():
        logger.error(f"Файл не найден: {INPUT_PATH}")
        return

    model = load_model()
    db_df = load_db_hack(INPUT_PATH, 'db_hack')
    db_texts, faiss_index = build_index(db_df, model)

    logger.info(f"Загрузка листа data из {INPUT_PATH}...")
    data_df = pd.read_excel(INPUT_PATH, sheet_name='data')
    logger.info(f"Загружено {len(data_df)} товаров для матчинга")

    if 'name' not in data_df.columns:
        logger.error("Колонка 'name' не найдена в листе data")
        return

    start_time = time.time()
    codes, scores = [], []
    matched_n = 0
    total = len(data_df)

    for i, (_, row) in enumerate(data_df.iterrows()):
        name = str(row['name'])
        idx, score = match_one(name, model, db_df, db_texts, faiss_index)

        if idx >= 0:
            codes.append(db_df.iloc[idx]['code'])
            scores.append(round(score, 4))
            matched_n += 1
        else:
            codes.append('')
            scores.append(0.0)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            logger.info(
                f"Прогресс: {i+1}/{total} ({100*(i+1)/total:.1f}%) | "
                f"Найдено: {matched_n} | "
                f"t={elapsed:.1f}s | ~{(elapsed/(i+1)):.3f}s/row"
            )

    result_df = data_df.copy()
    result_df['code'] = codes
    result_df['score'] = scores

    logger.info(f"Сохранение в {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='data', index=False)
        db_df.to_excel(writer, sheet_name='db_hack', index=False)

    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"ГОТОВО: {total} строк, найдено {matched_n}")
    logger.info(f"Время: {total_time:.1f}s ({total_time/total:.4f}s/row)")
    logger.info(f"Результат: {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
