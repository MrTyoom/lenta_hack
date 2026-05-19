"""
Утилита для очистки VRAM
Использовать в Jupyter notebook:

from vram_cleanup import cleanup_vram, auto_cleanup, VRAMGuard, log_vram_usage
cleanup_vram()
"""
import gc
import torch
import functools
from typing import Callable, Any

def log_vram_usage(label: str = ""):
    """Логирование использования VRAM"""
    if not torch.cuda.is_available():
        if label:
            print(f"[{label}] CUDA не доступна")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    
    if label:
        print(f"VRAM [{label}]: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")
    else:
        print(f"VRAM: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")

def cleanup_vram(verbose=False):
    """
    Очистка VRAM после тяжёлых операций
    
    Args:
        verbose: Выводить информацию о памяти
    """
    # Считаем память до очистки
    if verbose and torch.cuda.is_available():
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        reserved_before = torch.cuda.memory_reserved() / 1024**3
    
    # Очистка
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Считаем память после очистки
    if verbose and torch.cuda.is_available():
        allocated_after = torch.cuda.memory_allocated() / 1024**3
        reserved_after = torch.cuda.memory_reserved() / 1024**3
        
        print("=" * 60)
        print("ОЧИСТКА VRAM ВЫПОЛНЕНА")
        print("=" * 60)
        print(f"До очистки:")
        print(f"  Allocated: {allocated_before:.2f} GB")
        print(f"  Reserved:  {reserved_before:.2f} GB")
        print(f"После очистки:")
        print(f"  Allocated: {allocated_after:.2f} GB (освобождено: {allocated_before - allocated_after:.2f} GB)")
        print(f"  Reserved:  {reserved_after:.2f} GB (освобождено: {reserved_before - reserved_after:.2f} GB)")
        print("=" * 60)

def auto_cleanup(func: Callable) -> Callable:
    """
    Декоратор для автоматической очистки VRAM после выполнения функции
    
    Usage:
        @auto_cleanup
        def train_model():
            # heavy computation
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            cleanup_vram(verbose=False)
    return wrapper

class VRAMGuard:
    """
    Контекстный менеджер для очистки VRAM до и после блока кода
    
    Usage:
        with VRAMGuard("model training"):
            # heavy computation
            pass
    """
    def __init__(self, label: str = ""):
        self.label = label
    
    def __enter__(self):
        if self.label:
            log_vram_usage(f"BEFORE {self.label}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_vram(verbose=False)
        if self.label:
            log_vram_usage(f"AFTER {self.label}")
        return False

def get_vram_info():
    """Вывод информации о использовании VRAM"""
    if not torch.cuda.is_available():
        print("CUDA не доступна")
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print("=" * 60)
    print("ИНФОРМАЦИЯ О VRAM")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Текущая выделенная память: {allocated:.2f} GB")
    print(f"  Текущая зарезервированная память: {reserved:.2f} GB")
    print(f"  Максимальная выделенная (за сессию): {max_allocated:.2f} GB")
    print("=" * 60)

def deep_cleanup(*objects):
    """
    Глубокая очистка: удаление объектов + очистка VRAM
    
    Usage:
        deep_cleanup(model1, model2, large_array)
    """
    for obj in objects:
        if obj is not None:
            del obj
    cleanup_vram(verbose=False)

if __name__ == "__main__":
    cleanup_vram()
    get_vram_info()
