"""
API модуль для обработки аудиофайлов.

Данный модуль предоставляет REST API для загрузки и обработки аудиофайлов.
Включает в себя эндпоинты для обработки аудио с применением шумоподавления,
фильтрации и нормализации.
"""

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.config.config import MAX_FILE_SIZE, LOGS_DIR, INPUT_DIR, OUTPUT_DIR
from app.src.audio_processor import process_audio_pipeline
from app.src.exceptions import (
    AudioLoadError,
    AudioSaveError,
    AudioFormatError,
    AudioParameterError,
    NSNet2EnhancerError,
    FilterError,
    NormalizationError,
    AudioProcessingError
)


router = APIRouter()

# Настройка базового конфига
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Создание необходимых директорий
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(LOGS_DIR).mkdir(exist_ok=True)


def map_audio_exception_to_http(ex: AudioProcessingError) -> HTTPException:
    """
    Функция для преобразования исключений обработки аудио в HTTP исключения.

    Args:
        ex : AudioProcessingError
            Исключение типа AudioProcessingError или его наследников.

    Returns:
        HTTPException: Соответствующее HTTP исключение с подходящим статус-кодом.

    Example:
        >> audio_ex = AudioFormatError("Неверный формат")
        >> http_ex = map_audio_exception_to_http(audio_ex)
        >> print(http_ex.status_code)
        >> 400
    """
    if isinstance(ex, (AudioFormatError, AudioParameterError)):
        return HTTPException(status_code=400, detail=str(ex))

    elif isinstance(ex, (AudioLoadError, AudioSaveError, NSNet2EnhancerError,
                         FilterError, NormalizationError)):
        return HTTPException(status_code=500, detail=str(ex))

    else:
        return HTTPException(
            status_code=500,
            detail=f"Неизвестная ошибка обработки: {str(ex)}"
        )


def validate_uploaded_file(file: UploadFile):
    """
    Функция для проверки загруженного файла на соответствие требованиям.

    Args:
        file : UploadFile
            Загруженный файл для валидации.

    Raises:
        HTTPException: Если файл не соответствует требованиям:
            - 400: Неподдерживаемый тип контента.
            - 400: Неподдерживаемое расширение файла.
            - 400: Превышен максимальный размер файла.

    Example:
        >> file = UploadFile(filename="test.wav", content_type="audio/wav")
        >> validate_uploaded_file(file)  # Не вызовет исключение
    """
    supported_content_types = ["audio/wav", "audio/mp3", "audio/mpeg"]
    if file.content_type not in supported_content_types:
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только форматы WAV и MP3"
        )

    if not file.filename.lower().endswith((".wav", ".mp3")):
        raise HTTPException(
            status_code=400,
            detail="Файл имеет неподдерживаемый формат"
        )

    if file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Файл имеет размер более 50 Мб"
        )


def get_file_extension(filename: str) -> str:
    """
    Функция для определения расширения файла на основе имени.

    Args:
        filename : str
            Имя файла для анализа.

    Returns:
        str : str
            Расширение файла (".wav" или ".mp3").

    Example:
        >> get_file_extension("audio.mp3")
        '.mp3'
        >> get_file_extension("sound.WAV")
        '.wav'
    """
    if filename.lower().endswith(".mp3"):
        return ".mp3"
    return ".wav"


def cleanup_temp_file(file_path: str):
    """
    Функция для удаления временного файла с обработкой ошибок.

    Args:
        file_path : str
            Путь к файлу для удаления.

    Note:
        Функция не вызывает исключения, только логирует предупреждения в случае
        невозможности удаления файла
    """

    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Временный файл удалён: {file_path}")
        except Exception as cleanup_ex:
            logger.warning(f"Не удалось удалить временный файл {file_path}: {cleanup_ex}")


@router.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    """
    Функция для обработки загруженного аудиофайла через пайплайн обработки.

    Выполняет полный пайплайн обработки аудио:
    1. Валидация загруженного файла.
    2. Сохранение во временную директорию.
    3. Обработка через пайплайн (шумоподавление, фильтрация, нормализация).
    4. Возврат ссылки на обработанный файл.
    5. Очистка временных файлов.

    Args:
        file : UploadFile
            Загруженный аудиофайл в формате WAV или MP3.

    Returns:
        dict : dict
            Словарь с ключом 'processed_file', содержащий URL обработанного файла.

    Raises:
        HTTPException: В случае ошибок при валидации или обработке:
            - 400: Проблемы с форматом файла, размером или содержимым.
            - 500: Внутренние ошибки обработки или сохранения.

    Example:
        Успешный ответ:
        ```json
        {
            "processed_file": "http://127.0.0.1:8000/output/processed_90b99262-1cbc-4640-8a31-fdfcc3e526c3.wav"
        }
        ```
    """
    # Проверка формата файла
    validate_uploaded_file(file)

    # Определение расширения и создание пути файлов
    ext = get_file_extension(file.filename)
    file_id = str(uuid.uuid4())
    input_path = f"{INPUT_DIR}/uploaded_{file_id}{ext}"
    output_path = f"{OUTPUT_DIR}/processed_{file_id}{ext}"

    try:
        # Сохранение загруженного файла
        with open(input_path, "wb") as f:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Загружен пустой файл.")
            f.write(content)

        # Обработка файла в отдельном потоке
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(
                executor,
                process_audio_pipeline,
                input_path,
                output_path
            )

        # Формирование URL результата
        result_url = f"http://127.0.0.1:8000/output/processed_{file_id}{ext}"
        logger.info(f"Файл успешно обработан: {result_url}")

        return {"processed_file": result_url}

    except AudioProcessingError as audio_ex:
        # Преобразование исключений обработки аудио в HTTP
        logger.error(f"Ошибка обработки аудио: {type(audio_ex).__name__}: {str(audio_ex)}")
        http_ex = map_audio_exception_to_http(audio_ex)
        raise http_ex

    except HTTPException as http_ex:
        # Пропуск HTTP исключений как есть
        logger.error(f"HTTP ошибка: {http_ex.detail}")
        raise http_ex

    except Exception as ex:
        # Обработка непредвиденных ошибок
        logger.error(f"Неожиданная ошибка: {type(ex).__name__}: {str(ex)}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    finally:
        # Очистка временных файлов
        cleanup_temp_file(input_path)