"""
Скрипт для добавления ячеек очистки VRAM после каждой code-cell в notebook
"""
import json
import sys

NOTEBOOK_PATH = "main.ipynb"
OUTPUT_PATH = "main_with_cleanup.ipynb"

# Код ячейки очистки VRAM
CLEANUP_CODE = """# === ОЧИСТКА VRAM ===
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Проверка памяти
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")"""

def add_cleanup_cells():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    code_cells_count = 0
    
    for i, cell in enumerate(nb['cells']):
        new_cells.append(cell)
        
        # Добавляем очистку после каждой code-cell
        if cell['cell_type'] == 'code':
            code_cells_count += 1
            
            # Не добавляем очистку после импортов и конфига
            source = ''.join(cell.get('source', []))
            skip_cleanup = any([
                'import cv2' in source and 'from pathlib' in source,  # Cell 1 - импорты
                'config = OmegaConf.load' in source,  # Cell 2 - конфиг
            ])
            
            if not skip_cleanup:
                cleanup_cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'id': f'cleanup_after_{code_cells_count}',
                    'metadata': {},
                    'outputs': [],
                    'source': CLEANUP_CODE.split('\n')[:-1]  # Убираем последнюю пустую строку
                }
                # Форматируем как список строк
                cleanup_cell['source'] = [line + '\n' for line in CLEANUP_CODE.split('\n') if line]
                cleanup_cell['source'][-1] += '\n'
                new_cells.append(cleanup_cell)
                print(f"Добавлена очистка после code-cell #{code_cells_count}")
    
    # Сохраняем новый notebook
    nb['cells'] = new_cells
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"\nГотово! Создан файл: {OUTPUT_PATH}")
    print(f"Всего code-cells: {code_cells_count}")
    print(f"Добавлено ячеек очистки: {code_cells_count - 2}")  # Минус импорты и конфиг

if __name__ == "__main__":
    add_cleanup_cells()
