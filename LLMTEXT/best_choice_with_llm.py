import os
import re
import pandas as pd
import ollama
import logging

INPUT_FILE = "result.xlsx"
MODEL_NAME = "qwen2.5:7b"
LOG_FILE = "skumatcher.log"


class OllamaSKUMatcher:
    def __init__(self, model_name: str, log_file: str = "skumatcher.log"):
        self.model_name = model_name
        self._setup_logging(log_file)

    def _setup_logging(self, log_file: str):
        """
        Настраивает логирование одновременно в консоль и в текстовый файл с поддержкой UTF-8.
        """
        self.logger = logging.getLogger("SKUMatcher")
        self.logger.setLevel(logging.INFO)
        
        # Очищаем старые хендлеры, если они были (защита от дублирования логов)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Формат вывода: [Время] [Уровень] Сообщение
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Хендлер для записи в файл (всегда в UTF-8)
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Хендлер для вывода в консоль
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

    @staticmethod
    def clean_sku(val) -> str:
        """
        Очищает и форматирует SKU, предотвращая считывание целых чисел как float (123.0).
        """
        if pd.isna(val):
            return ""
        if isinstance(val, float):
            if val.is_integer():
                return str(int(val))
            return str(val)
        return str(val).strip()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Универсально загружает данные из файлов .json, .xlsx или .csv.
        """
        self.logger.info(f"Загрузка файла данных: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден!")

        if file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError("Поддерживаются только форматы .csv, .json и .xlsx")

    def prepare_candidates(self, row: pd.Series) -> tuple[str, list]:
        """
        Собирает текстовое представление топ-5 кандидатов и список их SKU.
        """
        candidates_text = ""
        valid_skus = []
        
        for i in range(1, 6):
            name = row.get(f'top{i}', '')
            sku = self.clean_sku(row.get(f'top{i}_sku', ''))
            
            if name or sku:
                candidates_text += f"{i}. Товар: {name} | SKU: {sku}\n"
                if sku:
                    valid_skus.append(sku)
                    
        return candidates_text, valid_skus

    @staticmethod
    def build_prompt(raw_text: str, candidates_text: str, valid_skus: list) -> str:
        """
        Формирует строгий промпт для языковой модели с акцентом на непрямое совпадение,
        ключевые слова, объемы и характеристики товара.
        """
        example_sku = valid_skus[0] if valid_skus else "123456"

        prompt = f"""Ты — эксперт по анализу данных в ритейле. Твоя задача — сопоставить искаженный и зашумленный текст, распознанный с магазинного ценника (raw_text), с наиболее подходящим реальным товаром из базы данных (список кандидатов).

    ### ПРАВИЛА АНАЛИЗА И ЗАЩИТЫ ОТ ОШИБОК
    1. **ГЛУБОКИЙ АНАЛИЗ ЗАГЛУШЕК**: В raw_text могут присутствовать названия колонок ("Продукты_название", "Цена_без_карты") вместе с текстовым разбором ценника. 
       - Если в этом описании упомянут РЕАЛЬНЫЙ ТОВАР (например: *Название продукта: "Огурец"* или *Название продукта: "Сметана"*), ты ОБЯЗАН использовать это слово как главный ориентир для поиска в базе кандидатов!
       - Если же в тексте идут ТОЛЬКО названия колонок и фраза "Обратите внимание, что некоторые поля видны/не видны..." и НЕТ никакого конкретного названия товара — только тогда считай строку пустой и выводи `None`.
    2. **ЗАПРЕТ ГАЛЛЮЦИНАЦИЙ**: Использовать в <thinking> только те бренды, товары и ключевые слова, которые прямо написаны в текущем raw_text. Запрещено выдумывать посторонние бренды (например, SNICKERS), если их нет в тексте.
    3. **ПРАВИЛО ИСТИННОГО ИСКЛЮЧЕНИЯ**: Если в процессе анализа методом исключения ты выясняешь, что ВСЕ кандидаты ложные (не совпадают по бренду, типу товара или категории), ты КАТЕГОРИЧЕСКИ НЕ ИМЕЕШЬ ПРАВА выбирать какой-либо SKU из списка. Единственно верный ответ в этом случае — None.
    4. **ЗАПРЕТ НА ЛОЖНЫЙ ВЫВОД**: Запрещено писать фразы в стиле "Таким образом, наиболее подходящий товар — [чужой товар]". Если совпадений нет, финальный вывод в рассуждениях должен быть строго: "Совпадений не обнаружено, все кандидаты ложные".
    5. **БЕЗОПАСНОСТЬ ДАННЫХ**: Не придумывай характеристики товаров и не связывай SKU продуктов питания (йогурт, мармелад) с непродовольственными товарами или алкоголем.

    КОНТЕКСТ И ОСОБЕННОСТИ:
    1. В raw_text много мусора: технические символы (<0x0A>), цены, даты, штрихкоды, проценты скидок (например, -26%).
    2. Прямого и точного названия товара в raw_text может НЕ БЫТЬ. Оно часто обрезано или искажено.
    3. Опирайся на КЛЮЧЕВЫЕ СЛОВА (бренд, тип товара, вкус, цвет) и ХАРАКТЕРИСТИКИ:
    - Объем / Вес / Размер (например: 0.5л, 500г, 0.75, 1кг). Это критически важно для точного матчинга.
    - Жирность (для молочных продуктов), крепость или сорт (для алкоголя).
    - Количество штук в упаковке.

    ФОРМАТ ОТВЕТА:
    Ты должен ответить СТРОГО в следующем формате (сначала размышление, потом SKU):
    
    ### ПРАВИЛА АНАЛИЗА И МЫШЛЕНИЯ:
    В блоке `<thinking>` ты обязан строго следовать трем шагам:
    1. Выделить из raw_text все ключевые слова: бренд, тип товара, цвет/вкус, объем, размер, крепость/жирность, отсекая мусор (цены, даты).
    2. Выписать ключевые отличия кандидатов из списка топ-5.
    3. Провести сравнение методом исключения: объяснить, почему другие кандидаты НЕ подходят (не тот бренд, не тот цвет, не тот объем) и почему выбран именно этот товар.

    В блоке `<result>` не пиши никаких объяснений, вводных слов, знаков препинания, пробелов или форматирования. СЮДА ПИШИ ТОЛЬКО SKU (например, {example_sku}) ИЛИ None
    
    ### ТЕКУЩЕЕ ЗАДАНИЕ ДЛЯ ОБРАБОТКИ:
    Текст с ценника (raw_text):
    \"\"\"{raw_text}\"\"\"

    Список кандидатов (топ-5 из БД):
    {candidates_text}
    
    Выдай ответ строго в формате указанного выше образца (используй теги <thinking> для рассуждения и <result> для финального SKU или None)"""

        return prompt

    def query_ollama(self, prompt: str, valid_skus: list) -> tuple[str, str]:
        """
        Выполняет запрос к Ollama. 
        Возвращает кортеж: (найденный_sku, текст_размышления).
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.0}
            )
            full_response = response.get('response', '').strip()
            
            # Извлекаем размышления ИИ из тегов <thinking>
            thinking_match = re.search(r'<thinking>\s*(.*?)\s*</thinking>', full_response, re.DOTALL)
            thinking_text = thinking_match.group(1).strip() if thinking_match else "Размышления не найдены в ответе модели."
            
            # Извлекаем финальный результат из тегов <result>
            result_match = re.search(r'<result>\s*(.*?)\s*</result>', full_response, re.DOTALL)
            llm_result = result_match.group(1).strip() if result_match else full_response

            # Валидация ответа
            found_skus = [sku for sku in valid_skus if sku in llm_result]
            if found_skus:
                return found_skus[0], thinking_text
                
            if "none" in llm_result.lower():
                return "None", thinking_text
                
            # Запасной поиск цифр, если модель нарушила формат
            digits = re.findall(r'\d+', llm_result)
            if digits and digits[0] in valid_skus:
                return digits[0], thinking_text
                
            return "None", thinking_text
            
        except Exception as e:
            return "Error", f"Ошибка при вызове Ollama: {e}"

    def process_dataset(self, file_path: str) -> dict:
        """
        Основной цикл обработки датасета. Выводит логи размышлений на каждой строке.
        """
        df = self.load_data(file_path)
        row_to_sku = {}
        total_rows = len(df)
        
        self.logger.info(f"Начало обработки набора данных ({total_rows} строк). Модель: {self.model_name}")
        
        for idx, row in df.iterrows():
            # Номер строки для отображения пользователю (начиная с 1)
            display_row_num = idx + 1 
            raw_text = str(row.get('raw_text', ''))
            
            candidates_text, valid_skus = self.prepare_candidates(row)
            
            if not valid_skus:
                row_to_sku[idx] = "None"
                log_msg = (
                    f"\n--- [Строка {display_row_num}/{total_rows}] ---\n"
                    f"Кандидаты с SKU отсутствуют в исходных данных.\n"
                    f"🎯 ИТОГОВЫЙ SKU -> None\n"
                    f"{'='*60}"
                )
                self.logger.info(log_msg)
                continue
                
            prompt = self.build_prompt(raw_text, candidates_text, valid_skus)
            final_sku, thinking_process = self.query_ollama(prompt, valid_skus)
            
            row_to_sku[idx] = final_sku
            
            short_raw_text = raw_text[:120] + "..." if len(raw_text) > 120 else raw_text
            log_msg = (
                f"\n--- [Строка {display_row_num}/{total_rows}] ---\n"
                f"RAW_TEXT: {short_raw_text}\n\n"
                f"🧠 РАЗМЫШЛЕНИЯ МОДЕЛИ:\n{thinking_process}\n\n"
                f"🎯 ИТОГОВЫЙ SKU -> {final_sku}\n"
                f"{'='*60}"
            )
            self.logger.info(log_msg)
            
        return row_to_sku


def main():
    try:
        # Инициализируем наш класс с указанием файла логов
        matcher = OllamaSKUMatcher(model_name=MODEL_NAME, log_file=LOG_FILE)
        
        # Запускаем пайплайн
        result_dict = matcher.process_dataset(INPUT_FILE)
        
        matcher.logger.info("\n=== ОБРАБОТКА ПОЛНОСТЬЮ ЗАВЕРШЕНА ===")
        matcher.logger.info(f"Словарь результатов сохранен в лог-файл {LOG_FILE}")
        matcher.logger.info(f"Результаты: {result_dict}")

    except Exception as e:
        print(f"Критическая ошибка в main: {e}")


if __name__ == "__main__":
    main()