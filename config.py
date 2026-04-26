import os
from pathlib import Path

#Структура директорий
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
ANNOTATIONS_FILE: str = os.path.join(DATA_DIR, "annotations.json")
MODEL_OUTPUT_DIR: str = os.path.join(BASE_DIR, "my_massbert_model")

# Базовые параметры NLP-конвейера
LANG: str = "ru"
PIPELINE: list[str] = ["spancat"]
SPANS_KEY: str = "sc"

# Гиперпараметры машинного обучения
TRAIN_SPLIT: float = 0.8
SEED: int = 42
N_ITER: int = 50
DROPOUT: float = 0.3

# Настройки динамического пакетирования
BATCH_START: float = 4.0
BATCH_STOP: float = 32.0
BATCH_COMPOUND: float = 1.001