import os
import re
import torch
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def clean_ocr_text(text: str) -> str:
    """
    Очистка OCR текста от мусора.
    Оставляет только ключевые слова для LLM.
    
    Удаляет:
    - XML-подобные теги: <0x0A>, <F0>...
    - Специальные символы: _, ▁, emoji
    - Технические символы: <, >, hex-коды
    - Цены, даты, баркоды
    
    Оставляет:
    - Буквы (русские + латиница)
    - Цифры (только в составе слов)
    - Проценты, точки, дефисы (для объемов: 0.5л, 3.2%)
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Удаление XML-тегов и hex-кодов
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'0x[0-9A-Fa-f]+', ' ', text)
    
    # Удаление специальных символов
    text = text.replace('_', ' ')
    text = text.replace('▁', ' ')
    
    # Оставляем буквы, цифры, %, ., - (для объемов)
    text = re.sub(r'[^\w\sа-яА-Яa-zA-Z0-9.,%-]', ' ', text)
    
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Удаляем короткие слова (1-2 символа) - это обычно мусор
    words = [w for w in text.split() if len(w) > 2]
    
    return ' '.join(words).strip()


class HFSKUMatcher:
    """
    Hugging Face LLM matcher для выбора SKU из топ-5 кандидатов.
    Использует Qwen2.5-7B-Instruct с 4-bit квантованием для экономии VRAM.
    """
    
    DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 8,
        max_new_tokens: int = 64,
        temperature: float = 0.1,
        log_to_file: bool = False,
        log_file: str = "hf_sku_matcher.log"
    ):
        """
        Инициализация HF SKU Matcher.
        
        Args:
            model_name: Название модели на Hugging Face
            batch_size: Размер пакета для batch inference
            max_new_tokens: Максимальное количество токенов в ответе
            temperature: Температура генерации (0.1 для детерминированности)
            log_to_file: Логировать в файл (False = только консоль)
            log_file: Путь к файлу логов
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self._setup_logging(log_to_file, log_file)
        self.model = None
        self.tokenizer = None
        
    def _setup_logging(self, log_to_file: bool, log_file: str):
        """Настройка логирования"""
        self.logger = logging.getLogger("HFSKUMatcher")
        self.logger.setLevel(logging.INFO)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        if log_to_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)
    
    def load_model(self):
        """Загрузка модели с 4-bit квантованием"""
        if self.model is not None:
            self.logger.info("Модель уже загружена")
            return
        
        self.logger.info(f"Загрузка модели: {self.model_name}")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        
        self.logger.info(f"Модель загружена: {self.model_name}")
        self.logger.info(f"Устройство: {next(self.model.parameters()).device}")
    
    @staticmethod
    def clean_sku(val) -> str:
        """Очистка SKU от артефактов"""
        if pd.isna(val):
            return ""
        if isinstance(val, float):
            if val.is_integer():
                return str(int(val))
            return str(val)
        return str(val).strip()
    
    def prepare_candidates(self, row: pd.Series) -> Tuple[str, List[str]]:
        """
        Подготовка текста кандидатов из топ-5.
        
        Returns:
            Tuple с текстом кандидатов и списком валидных SKU
        """
        candidates_text = ""
        valid_skus = []
        
        for i in range(1, 6):
            name = row.get(f'top{i}', '')
            sku = self.clean_sku(row.get(f'top{i}_sku', ''))
            
            if name or sku:
                candidates_text += f"{i}. {name} | SKU: {sku}\n"
                if sku:
                    valid_skus.append(sku)
        
        return candidates_text, valid_skus
    
    def build_prompt(self, raw_text: str, candidates_text: str) -> str:
        """
        Построение промпта для LLM.
        Концептуальный поиск с учетом ошибок OCR.
        """
        # Очистка текста
        cleaned_text = clean_ocr_text(raw_text)
        
        prompt = f"""Ты эксперт по анализу данных ритейла. Сопоставь зашумленный OCR текст с товаром из базы.

ВАЖНО:
1. OCR содержит ошибки и мусор - ищи только КЛЮЧЕВЫЕ СЛОВА
2. Ключевые слова: бренд (CRISPY, WINE, GUSTARE), тип товара (вино, молоко, хлеб), объем (1л, 500г)
3. НЕ требуй точного совпадения - ищи концептуальное сходство
4. Примеры ошибок OCR: "GRILL" вместо "CRISPY", "WINE" вместо "VINE"

ПРАВИЛА ВЫБОРА:
- Если в OCR есть "WINE" → выбирай вино из кандидатов
- Если в OCR есть "MILK" или "МОЛОКО" → выбирай молоко
- Если в OCR есть бренд → выбирай товар этого бренда
- Смотри на объем/вес: 1л, 500г, 0.75л
- Если совсем ничего не понятно → None

ФОРМАТ ОТВЕТА СТРОГО:
<thinking>какие ключевые слова нашел, какой кандидат ближе по смыслу</thinking>
<result>SKU или None</result>

ПРИМЕР 1:
OCR: "GRILL WINE" (ошибка OCR, должно быть CRISPY)
Кандидаты: 1. Вино CRISPY WINE белое | SKU: 123456
Ответ: <thinking>OCR ошибся - "GRILL" похоже на "CRISPY". Оба "WINE" - вино. Кандидат 1 подходит.</thinking>
<result>123456</result>

ПРИМЕР 2:
OCR: "CHOCOLATE MILK 1L"
Кандидаты: 1. Молоко шоколадное 1л | SKU: 234567
Ответ: <thinking>В OCR "CHOCOLATE MILK" - шоколадное молоко, объем 1л. Кандидат 1 подходит.</thinking>
<result>234567</result>

ПРИМЕР 3:
OCR: "печенье овсяное"
Кандидаты: 1. Вино красное | SKU: 345678
Ответ: <thinking>В OCR печенье, в кандидатах вино. Совпадений нет.</thinking>
<result>None</result>

ТЕКУЩЕЕ ЗАДАНИЕ:
RAW_TEXT (очищенный): {cleaned_text}

КАНДИДАТЫ (топ-5):
{candidates_text}

Ответ (сначала <thinking>, потом <result>):"""
        return prompt
    
    def parse_response(self, response: str, valid_skus: List[str]) -> Tuple[str, str]:
        """
        Парсинг ответа модели.
        
        Returns:
            Tuple с финальным SKU и текстом размышлений
        """
        # Извлекаем размышления из тегов <thinking>
        thinking_match = re.search(r'<thinking>\s*(.*?)\s*</thinking>', response, re.DOTALL | re.IGNORECASE)
        thinking_text = thinking_match.group(1).strip() if thinking_match else "Нет размышлений"
        
        # Извлекаем результат из тегов <result>
        result_match = re.search(r'<result>\s*(.*?)\s*</result>', response, re.DOTALL | re.IGNORECASE)
        result_text = result_match.group(1).strip() if result_match else ""
        
        # Если теги не найдены, пробуем найти результат в конце ответа
        if not result_text:
            # Ищем SKU в последней строке или после слова "result"
            lines = response.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('<'):
                    result_text = line
                    break
        
        # Поиск SKU в result_text
        found_skus = [sku for sku in valid_skus if sku in result_text]
        if found_skus:
            return found_skus[0], thinking_text
        
        # Проверка на None
        if "none" in result_text.lower() or result_text.strip() == "" or result_text.strip().lower() == "none":
            return "None", thinking_text
        
        # Запасной вариант: ищем первое число в result_text
        digits = re.findall(r'\d+', result_text)
        if digits and digits[0] in valid_skus:
            return digits[0], thinking_text
        
        # Если ничего не найдено - возвращаем None
        return "None", thinking_text
    
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Генерация ответов для пакета промптов"""
        if self.model is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model()")
        
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        responses = []
        for i, prompt in enumerate(prompts):
            generated = self.tokenizer.decode(
                outputs[i][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            responses.append(generated.strip())
        
        return responses
    
    def process_row(self, row: pd.Series, total_rows: int, row_idx: int) -> Tuple[str, str]:
        """Обработка одной строки dataframe"""
        raw_text = str(row.get('raw_text', row.get('ocr_text', '')))
        candidates_text, valid_skus = self.prepare_candidates(row)
        
        if not valid_skus:
            return "None", "Нет валидных SKU в кандидатах"
        
        prompt = self.build_prompt(raw_text, candidates_text)
        response = self.generate_batch([prompt])[0]
        final_sku, thinking = self.parse_response(response, valid_skus)
        
        short_raw = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
        self.logger.info(f"[{row_idx+1}/{total_rows}] RAW: {short_raw} -> SKU: {final_sku}")
        
        return final_sku, thinking
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        ocr_col: str = 'ocr_text',
        output_col: str = 'id_sku'
    ) -> pd.DataFrame:
        """
        Обработка dataframe с топ-5 кандидатами.
        
        Args:
            df: DataFrame с колонками ocr_text/raw_text, top1-top5, top1_sku-top5_sku
            ocr_col: Название колонки с OCR текстом
            output_col: Название колонки для результата
        
        Returns:
            DataFrame с добавленной колонкой id_sku
        """
        self.load_model()
        
        total_rows = len(df)
        self.logger.info(f"Начало обработки {total_rows} строк")
        
        results = []
        
        for idx in tqdm(range(0, total_rows, self.batch_size), desc="Обработка"):
            batch_end = min(idx + self.batch_size, total_rows)
            batch_df = df.iloc[idx:batch_end]
            
            batch_prompts = []
            batch_indices = []
            batch_valid_skus = []
            
            for local_idx, (_, row) in enumerate(batch_df.iterrows()):
                global_idx = idx + local_idx
                
                # Получаем raw_text и очищаем
                raw_text = str(row.get(ocr_col, row.get('raw_text', '')))
                cleaned_text = clean_ocr_text(raw_text)
                
                candidates_text, valid_skus = self.prepare_candidates(row)
                
                if not valid_skus:
                    results.append({
                        'row_idx': global_idx,
                        'sku': "None",
                        'thinking': "Нет валидных SKU"
                    })
                else:
                    # Используем очищенный текст для промпта
                    prompt = self.build_prompt(cleaned_text, candidates_text)
                    batch_prompts.append(prompt)
                    batch_indices.append(global_idx)
                    batch_valid_skus.append(valid_skus)
            
            if batch_prompts:
                batch_responses = self.generate_batch(batch_prompts)
                
                for local_idx, (response, valid_skus) in enumerate(zip(batch_responses, batch_valid_skus)):
                    global_idx = batch_indices[local_idx]
                    final_sku, thinking = self.parse_response(response, valid_skus)
                    
                    results.append({
                        'row_idx': global_idx,
                        'sku': final_sku,
                        'thinking': thinking
                    })
        
        result_df = df.copy()
        sku_list = [None] * len(result_df)
        thinking_list = [None] * len(result_df)
        
        for r in results:
            sku_list[r['row_idx']] = r['sku'] if r['sku'] != "None" else None
            thinking_list[r['row_idx']] = r['thinking']
        
        result_df[output_col] = sku_list
        result_df['llm_thinking'] = thinking_list
        
        self.logger.info(f"Обработка завершена. Найдено SKU: {sum(1 for s in sku_list if s is not None)}/{total_rows}")
        
        return result_df
    
    def unload_model(self):
        """Выгрузка модели для освобождения VRAM"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.logger.info("Модель выгружена")


def main():
    """Тестовый запуск"""
    test_data = {
        'ocr_text': [
            'молоко домик в деревне 3 2 цена 89 руб',
            'хлеб бородинский 500г акция -26%',
            'огурцы свежие весовые россия',
        ],
        'top1': ['Молоко Домик в деревне 3.2% 1л', 'Хлеб Бородинский 500г', 'Огурцы свежие 1кг'],
        'top1_sku': ['123456', '234567', '345678'],
        'top2': ['Молоко Простоквашино 3.2% 1л', 'Хлеб Дарницкий 500г', 'Помидоры свежие 1кг'],
        'top2_sku': ['456789', '567890', '678901'],
    }
    
    df_test = pd.DataFrame(test_data)
    
    matcher = HFSKUMatcher(batch_size=2)
    result = matcher.process_dataframe(df_test)
    
    print("\n=== РЕЗУЛЬТАТЫ ===")
    print(result[['ocr_text', 'id_sku', 'llm_thinking']])


if __name__ == "__main__":
    main()
