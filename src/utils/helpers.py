# src/utils/helpers.py

import datetime
import numpy as np
import os
import random
import time
import importlib

def set_global_seed(seed=42):
    """Define seeds para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

def current_datetime(fmt='%Y-%m-%d %H:%M:%S'):
    """Retorna data/hora atual formatada."""
    return datetime.datetime.now().strftime(fmt)

class Timer:
    """Context manager para medir tempo de execução."""
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.time() - self.start
        if self.name:
            print(f'{self.name} levou {elapsed:.3f}s')
        else:
            print(f'Tempo decorrido: {elapsed:.3f}s')

def check_dependency(pkg):
    """Verifica se pacote está instalado."""
    spec = importlib.util.find_spec(pkg)
    return spec is not None

def normalize_array(arr):
    """Normaliza array NumPy para intervalo [0,1]."""
    arr = np.asarray(arr)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)
