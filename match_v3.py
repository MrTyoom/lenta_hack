"""
Матчинг: data.name → db_hack.fullname
FAISS (bge-m3) + rapidfuzz.token_sort_ratio + Jaccard
Всегда top-1, без порога
"""
import re
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

SCRIPT_DIR = Path(__file__).parent.absolute()
LLMTEXT_DIR = SCRIPT_DIR / "LLMTEXT"
INPUT_PATH = SCRIPT_DIR / "match.xlsx"
OUTPUT_PATH = SCRIPT_DIR / "match_result.xlsx"
MODEL_PATH = str(LLMTEXT_DIR / "bge-m3")
MODEL_NAME = "deepvk/USER-bge-m3"
FAISS_CACHE = str(LLMTEXT_DIR / "faiss_match_v3.bin")
TEXTS_CACHE = str(LLMTEXT_DIR / "texts_match_v3.pkl")
TOP_K = 20

# ---- Транслитерация ----
CYR_TO_LAT = {
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


def _translit(text):
    return ''.join(CYR_TO_LAT.get(c, c) for c in text)


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _translit(text)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'\b\d+\s*(г|кг|мл|л|шт|w|ml|l|g|kg)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s*[xхXХ×]\b', '', text)
    text = re.sub(r'\b\d+\s*\*\s*\d+\s*(г|кг|мл|л)?\b', '', text)
    text = re.sub(r'\b(м/у|м\/у|ст/б|ст\/б|ж/б|ж\/б|п/э|п\/э|пэт|дой-пак|пауч|бан|кор|уп)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+[,.]?\d*\s*%', '', text)
    text = re.sub(r'-\d+%', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(w for w in text.lower().split() if len(w) > 1)
    return text


def load_model():
    if Path(MODEL_PATH).exists():
        print(f"Загрузка модели из {MODEL_PATH}")
        return SentenceTransformer(MODEL_PATH)
    print(f"Загрузка модели {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    return model


def get_index(db_df, model):
    if Path(FAISS_CACHE).exists() and Path(TEXTS_CACHE).exists():
        print("Загрузка кэшированного индекса...")
        with open(TEXTS_CACHE, 'rb') as f:
            texts = pickle.load(f)
        index = faiss.read_index(FAISS_CACHE)
        print(f"Индекс загружен: {len(texts)} текстов")
        return texts, index

    print("Нормализация db_hack...")
    texts = [normalize(str(n)) for n in db_df['fullname']]
    print("Эмбеддинги...")
    embeddings = model.encode(
        texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"FAISS индекс ({embeddings.shape[0]} векторов)...")
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)

    faiss.write_index(index, FAISS_CACHE)
    with open(TEXTS_CACHE, 'wb') as f:
        pickle.dump(texts, f)
    print(f"Индекс сохранён: {FAISS_CACHE}")
    return texts, index


def match_batch(queries, model, db_df, db_texts, faiss_index):
    q_norms = [normalize(str(q)) for q in queries]
    valid = [(i, q) for i, q in enumerate(q_norms) if q]

    if not valid:
        return [("", 0.0)] * len(queries)

    q_embs = model.encode(
        [q for _, q in valid], batch_size=64, normalize_embeddings=True
    )
    D, I = faiss_index.search(q_embs, TOP_K)

    results = [("", 0.0)] * len(queries)
    for vi, (orig_i, q) in enumerate(valid):
        q_tokens = set(q.split())
        best_code, best_score = "", -1.0
        for j in range(len(I[vi])):
            cand_i = I[vi][j]
            if cand_i == -1:
                continue
            cand_text = db_texts[cand_i]
            faiss_s = float(D[vi][j])
            token_s = fuzz.token_sort_ratio(q, cand_text) / 100.0
            cand_tokens = set(cand_text.split())
            jaccard = len(q_tokens & cand_tokens) / max(1, len(q_tokens | cand_tokens))
            combined = 0.3 * faiss_s + 0.4 * token_s + 0.3 * jaccard
            if combined > best_score:
                best_score = combined
                best_code = str(db_df.iloc[cand_i]['code'])
        code = "" if best_code in ('nan', 'None', '') else best_code
        results[orig_i] = (code, round(best_score, 4))
    return results


def main():
    t0 = time.time()
    print("=" * 50)
    print("МАТЧИНГ V3: match.xlsx data <-> db_hack")
    print("=" * 50)

    model = load_model()

    print("Загрузка db_hack...")
    db_df = pd.read_excel(INPUT_PATH, sheet_name='db_hack', dtype={'code': str})
    db_df['code'] = db_df['code'].str.replace(r'\.0$', '', regex=True)
    print(f"  {len(db_df)} строк")

    db_texts, faiss_index = get_index(db_df, model)

    print("Загрузка data...")
    data_df = pd.read_excel(INPUT_PATH, sheet_name='data')
    data_df = data_df.drop(columns=['code', 'score'], errors='ignore')
    total = len(data_df)
    print(f"  {total} строк")

    print("Матчинг...")
    codes, scores = [], []
    BATCH = 500
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        batch_names = data_df['name'].iloc[start:end].tolist()
        batch_res = match_batch(batch_names, model, db_df, db_texts, faiss_index)
        for c, s in batch_res:
            codes.append(c)
            scores.append(s)
        elapsed = time.time() - t0
        rate = elapsed / end
        eta = rate * (total - end)
        print(f"  {end}/{total} ({100*end/total:.1f}%) | {elapsed:.0f}s | ETA {eta:.0f}s")

    data_df['code'] = codes
    data_df['score'] = scores

    print(f"Сохранение {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        data_df.to_excel(writer, sheet_name='data', index=False)
        db_df.to_excel(writer, sheet_name='db_hack', index=False)

    matched = sum(1 for c in codes if c)
    total_time = time.time() - t0
    print("=" * 50)
    print(f"ГОТОВО: {total} строк, {matched} с кодом ({100*matched/total:.1f}%)")
    print(f"Время: {total_time:.0f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
