import os
from pathlib import Path

# Базовая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Директории данных
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
LOGS_DIR = BASE_DIR / "app" / "logs"

# Создаем директории, если их ещё нет
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Параметры обработки аудио
FILTER_LOW = 300.0
FILTER_HIGH = 3400.0
FILTER_ORDER = 6

MAX_FILE_SIZE = 50 * 1024 * 1024