"""
Скрипт для добавления мониторинга VRAM в первые ячейки notebook
"""
import json

NOTEBOOK_PATH = "main.ipynb"
OUTPUT_PATH = "main_with_monitoring.ipynb"

# Ячейка мониторинга VRAM - вставка после импортов
MONITORING_CODE = [
    "# === МОНИТОРИНГ VRAM ===\n",
    "from vram_cleanup import get_vram_info, cleanup_vram\n",
    "\n",
    "# Показать информацию о памяти\n",
    "get_vram_info()\n"
]

def add_monitoring():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Находим ячейку с импортами (должна быть первой code cell)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'import cv2' in source and 'from pathlib' in source:
                # Добавляем мониторинг после ячейки с импортами
                monitoring_cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'id': 'vram_monitoring',
                    'metadata': {},
                    'outputs': [],
                    'source': MONITORING_CODE
                }
                nb['cells'].insert(i + 1, monitoring_cell)
                print(f"Добавлен мониторинг VRAM после ячейки с импортами (cell #{i+1})")
                break
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Готово! Создан файл: {OUTPUT_PATH}")

if __name__ == "__main__":
    add_monitoring()
