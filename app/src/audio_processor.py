"""
Модуль для обработки звука, включающий в себя удаление шумов, фильтрацию по диапазону частот
и нормализацию громкости.

Данный модуль предоставляет функции для обработки аудиофайлов, включающий в себя:
1. Загрузку и проверку подлинности аудиофайлов
2. Шумоподавление с использованием noisereduce
3. Фильтрацию с использованием полосового фильтра Баттерворта
4. Нормализацию звука по пику
5. Сохранение аудиофайла
6. Полный пайплайн обработки звука

Пример использования:
>> process_audio("input.wav", "output.wav")
"""

import io
import logging

import numpy as np
import librosa
import librosa.util.exceptions
import magic
import soundfile as sf
from scipy import signal
from pydub import AudioSegment, effects
from nsnet2_denoiser import NSnet2Enhancer

import app.src.exceptions as app_exceptions
from app.config.config import FILTER_LOW, FILTER_HIGH, FILTER_ORDER


logger = logging.getLogger(__name__)


def is_valid_audio(file_path: str) -> bool:
    """
    Функция для проверки валидности типа файлов.
    Использует библиотеку python-magic для определения типа файлов на основе содержимого, а не на основе расширения.
    Благодаря этому обеспечивает более надежную проверку.

    Args:
        file_path : str
            Путь к файлу, который нужно проверить.

    Returns:
        True - если файл принадлежит к "audio", в противном случае False.

    Example:
        >> is_valid_audio("test.wav")
        True
        >> is_valid_audio("doc.txt")
        False
    """
    try:
        mime_type = magic.from_file(file_path, mime=True)
        return mime_type.startswith("audio/")
    except Exception as e:
        return False


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Функция для загрузки аудиофайла, возвращает данные аудио и частоту дискретизации.

    Поддерживает форматы WAV и MP3. Проверяет целостность файла и звуковые параметры

    Args:
        file_path : str
            Путь к файлу для загрузки

    Returns:
        Кортеж, содержащий:
            - data : np.ndarray
                Числовой массив представления аудиофайла.
                Форма:
                    (samples,) - для одноканального аудио.
                    (samples, channels) - для многоканального аудио.
            - sample_rate : int
                Частота дискретизации в Гц.

    Raises:
        AudioFormatError: Если файл является недопустимым аудиофайлом, сломан или поврежден.
        AudioParameterError: Если параметры звука неверны (например, нулевая частота дискретизации).
        AudioLoadError: При непредвиденных ошибках во время загрузки.

    Example:
        >> data, sr = load_audio("audio.wav")
        >> print(f"Загружено {data} сэмплов в {sr} Гц")
        Загружено 81000 сэмплов в 44100 Гц
    """
    logger.info(f"Загрузка аудио: {file_path}")

    try:
        # Проверка валидности с использованием PyMagic
        if not is_valid_audio(file_path):
            logger.error("PyMagic обнаружил, что файл сломан или не является аудиофайлом")
            raise app_exceptions.AudioFormatError("PyMagic обнаружил, что файл сломан или не является аудиофайлом")

        # Загрузка аудиофайла через soundfile
        data, sr = sf.read(file_path)

        # Проверка валидности загруженных данных
        if data is None or len(data) == 0:
            logger.error("Файл пуст или содержит некорректные аудиоданные.")
            raise app_exceptions.AudioFormatError("Файл пуст или содержит некорректные аудиоданные.")

        if sr is None or sr <= 0:
            logger.error("Некорректная частота дискретизации.")
            raise app_exceptions.AudioFormatError("Некорректная частота дискретизации.")

        logger.info(f"Аудио загружено. Длина: {len(data)}, Частота: {sr} Гц")
        return data, sr

    except (sf.LibsndfileError, app_exceptions.AudioFormatError) as ex:
        logger.error(f"Ошибка soundfile: {str(ex)}")
        raise app_exceptions.AudioFormatError(f"Файл поврежден или имеет неподдерживаемый формат: {str(ex)}")

    except Exception as ex:
        if "NoBackendError" in str(type(ex).__name__) or "NoBackendError" in str(ex):
            logger.error(f"Нет подходящего метода для чтения файла: {str(ex)}")
            raise app_exceptions.AudioFormatError("Файл поврежден, имеет неподдерживаемый формат или не является аудиофайлом")
        else:
            logger.error(f"Неожиданная ошибка при загрузке аудио: {type(ex).__name__}: {str(ex)}")
            raise app_exceptions.AudioLoadError(f"Неожиданная ошибка при загрузке: {str(ex)}")


def save_audio(data: np.ndarray, sr: int, file_path: str):
    """
    Функция для сохранения аудиоданных в файл в формате WAV или MP3
    Функция сохраняет в формате, переданном ей на вход.

    Args:
        data : np.ndarray
            Числовой массив представления аудиофайла.
            Форма:
                (samples,) - для одноканального аудио.
                (samples, channels) - для многоканального аудио.
        sr : int
            Частота дискретизации в Гц.
        file_path : str
            Путь к выходному файлу. Расширение определяет формат.

    Raises:
        AudioSaveError: Если сохранение файла невозможно по каким-либо причинам

    Example:
        >> save_audio('output.wav', audio_data, 44100)
        >> save_audio('output.mp3', stereo_data, 48000)
    """
    try:
        sf.write(file_path, data, sr)
        logger.info(f"Аудио успешно сохранено: {file_path}")
    except Exception as ex:
        logger.error(f"Ошибка при сохранении аудио: {str(ex)}")
        raise app_exceptions.AudioSaveError(f"Ошибка при сохранении аудио: {str(ex)}")


def remove_noise(data: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """
    Функция для удаления шумов в аудиофайле.
    Функция использует предобученную модель NSnet2Enhancer от Microsoft для удаления шумов.

    Args:
        data : np.ndarray
            Числовой массив представления аудиофайла.
            Форма:
                (samples,) - для одноканального аудио.
                (samples, channels) - для многоканального аудио.
        sr : int
            Частота дискретизации в Гц.

    Returns:
        data : np.ndarray
            Аудиоданные с пониженным уровнем шума. Имеют ту же форму, что и входные данные.
        sr : int
            Обновленная частота дискретизации в Гц (16кГц или 48кГц).

    Raises:
        AudioFormatError: Если звук содержит только нули или присутствуют NaN, бесконечные значения (сломанное аудио).
        NSNet2EnhancerError: Если удаление шумов не сработало из-за ошибки.

    Example:
        >> # Пример с одноканальным аудио
        >> clean_mono, sr = remove_noise(mono_noisy_audio, 44100)
        >> # Пример с многоканальным аудио
        >> clean_stereo, sr = remove_noise(stereo_noisy_audio, 48000)

    Note:
        Из-за особенностей NSNet2Enhancer частота дискретизации обновляется до 16кГц или до 48кГЦ
        в зависимости от того, к какой частоте ближе частота дискретизации исходного аудио.
    """
    logger.info("Начало удаления шума...")

    try:
        # Проверка целостности аудиоданных
        if np.all(data == 0) or np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.error("Аудио повреждено или содержит только нули (тишина)")
            raise app_exceptions.AudioFormatError("Аудио повреждено или содержит только нули (тишина)")

        logger.info(f"Исходная частота дискретизации: {sr}")
        logger.info(f"Форма данных: {data.shape}")
        logger.info(f"Исходный тип данных: {data.dtype}")

        target_srs = [16000, 48000]

        target_sr = 48000
        # Выбор частоты дискретизации, к которой хотим привести наше аудио (в зависимости от близости)
        if abs(sr - 16000) < abs(sr - 48000):
            target_sr = 16000

        # Обновление аудио в соответствии с новой частотой дискретизации
        if sr not in target_srs:
            logger.info(f"Преобразование частоты дискретизации с {sr} Гц до {target_sr} Гц")
            if data.ndim == 1:
                audio_data_resampled = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)
            else:
                audio_data_resampled = librosa.resample(y=data.T, orig_sr=sr, target_sr=target_sr).T

            current_sr = target_sr
            audio_to_process = audio_data_resampled

        else:
            logger.info("Частота дискретизации совместима, ресэмплинг не требуется")
            current_sr = sr
            audio_to_process = data

        logger.info(f"Частота дискретизации для удаления шума: {current_sr} Гц")

        enhancer = NSnet2Enhancer(fs=current_sr)

        if audio_to_process.ndim == 1:
            audio_to_process = audio_to_process[:, np.newaxis]
            is_mono = True
            logger.info("Обнаружен одноканальный аудиофайл")
        else:
            is_mono = False
            logger.info(f"Обнаружен многоканальный аудиофайл с {audio_to_process.shape[1]} каналами")

        num_channels = audio_to_process.shape[1]
        denoised_channels = []

        # Обработка аудиофайла
        for ch in range(num_channels):
            logger.info(f"Удаление шума на канале: {ch + 1} / {num_channels}...")
            channel_data = audio_to_process[:, ch]

            try:
                denoised_channel = enhancer(channel_data, current_sr)
                denoised_channels.append(denoised_channel)
            except Exception as ex:
                logger.error(f"Ошибка при обработке канала: {ch + 1}: {ex}")
                denoised_channels.append(channel_data)

        denoised_audio = np.column_stack(denoised_channels)

        if is_mono:
            denoised_audio = denoised_audio[:, 0]

        final_sr = current_sr
        return denoised_audio, final_sr

    except app_exceptions.AudioFormatError as ex:
        # Обработка исключений AudioFormatError так, как они есть сейчас
        raise ex
    except Exception as ex:
        logger.error(f"Ошибка при удалении шума: {str(ex)}")
        raise app_exceptions.NSNet2EnhancerError(f"Ошибка при удалении шума: {str(ex)}")


def check_bandpass_filter_parameters(lowcut: float, highcut: float, order: int, nyquist: float):
    """
    Вспомогательная функция для проверки параметров полосового фильтра Баттерворта.
    Проверяет, что параметры фильтра находятся в допустимых диапазонах и физически значимы.

    Args:
        lowcut : Нижняя частота среза в Гц
        highcut : Верхняя частота среза в Гц
        order : Порядок фильтрации (должен быть в диапазоне от 1 до 20)
        nyquist : Частота Найквиста (sample_rate / 2) в Гц

    Raises:
        ValueError: Если какой-либо из параметров находится в недопустимых значениях,
        появляется сообщение об ошибке с описанием

    Example:
        >> check_bandpass_filter_parameters(300, 3400, 6, 8000)  # Валидные значения
        >> check_bandpass_filter_parameters(-100, 3400, 6, 8000)  # Вызов ошибки ValueError
    """
    if lowcut < 0 or highcut < 0:
        raise ValueError("Границы фильтра должны быть положительными")

    if lowcut > highcut:
        raise ValueError("Нижняя граница фильтра должна быть меньше верхней")

    if highcut > nyquist:
        raise ValueError("Верхняя граница фильтра превышает предел Найквиста")

    if order <= 0:
        raise ValueError("Порядок фильтра должен быть положительным числом")

    if order > 20:
        raise ValueError("Слишком высокий порядок фильтра")


def apply_bandpass_filter(data: np.ndarray,
                          sr: int,
                          lowcut: float = FILTER_LOW,
                          highcut: float = FILTER_HIGH,
                          order: int = FILTER_ORDER) -> np.ndarray:
    """
    Функция для применения полосового фильтра Баттерворта к аудиоданным.
    Применяет полосовой фильтр Баттерворта, чтобы пропускать сигналы в заданной частоте,
    используя scipy.signal.filtfilt.

    Args:
        data : np.ndarray
            Числовой массив представления аудиофайла.
            Форма:
                (samples,) - для одноканального аудио.
                (samples, channels) - для многоканального аудио.
        sr : int
            Частота дискретизации в Гц.
        lowcut : float
            Нижняя частота среза в Гц. По умолчанию используется значение FILTER_LOW.
        highcut : float
            Верхняя частота среза в Гц. По умолчанию используется значение FILTER_HIGH.
        order : int
            Порядок фильтрации. Чем выше порядок, тем круче переход. По умолчанию используется значение FILTER_ORDER.

    Returns:
        data : np.ndarray
            Отфильтрованные аудиоданные по диапазону частот. Имеют ту же форму, что и входные данные.

    Raises:
        AudioParameterError: Если параметры фильтра неверны.
        FilterError: Если операция фильтрации не сработала из-за ошибки.

    Example:
        >> # Применение базового фильтра
        >> filtered = apply_bandpass_filter(audio_data, 44100)
        >> # Применение фильтра со своим диапазоном частот и порядком (300-3400 Hz)
        >> speech_filtered = apply_bandpass_filter(audio_data, 16000, 300, 3400, 6)

    Note:
        Использует фильтрацию по нулевой фазе (scipy.signal.filtfilt), обрабатывая данные как в прямом,
        так и в обратном направлении для устранения искажений.
    """
    logger.info(f"Применение полосового фильтра {lowcut} - {highcut} Гц...")

    try:
        nyquist = sr / 2

        # Проверка параметров фильтрации
        check_bandpass_filter_parameters(lowcut, highcut, order, nyquist)

        # Нормализация частот до частоты Найквиста
        low = lowcut / nyquist
        high = highcut / nyquist

        # Создание полосового фильтра Баттерворта
        b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)

        # Применение фильтра
        if len(data.shape) > 1:
            # Обработка мультиканального аудио
            filtered = np.zeros_like(data)
            for channel_idx in range(data.shape[1]):
                filtered[:, channel_idx] = signal.filtfilt(b, a, data[:, channel_idx])
        else:
            # Обработка одноканального аудио
            filtered = signal.filtfilt(b, a, data)

        logger.info("Фильтрация завершена")
        return filtered

    except ValueError as ex:
        logger.error(f"Ошибка параметров фильтрации: {str(ex)}")
        raise app_exceptions.AudioParameterError(f"Ошибка параметров фильтрации: {str(ex)}")
    except Exception as ex:
        logger.error(f"Ошибка при применении фильтра: {str(ex)}")
        raise app_exceptions.FilterError(f"Ошибка при применении фильтра: {str(ex)}")


def normalize_audio(data: np.ndarray, sr: int) -> np.ndarray:
    """
    Функция для нормализации громкости звука с помощью эффектов pydub.
    Применяет нормализацию пиков, чтобы максимально увеличить диапазон.
    Использует эффект нормализации pydub, который регулирует усиление таким образом, чтобы максимально громкий
    пик достигал около 0 ДБ.

    Args:
        data : np.ndarray
            Числовой массив представления аудиофайла.
            Форма:
                (samples,) - для одноканального аудио.
                (samples, channels) - для многоканального аудио.
        sr : int
            Частота дискретизации в Гц.

    Returns:
        data : np.ndarray
            Нормализованные по громкости аудиоданные. Имеют ту же форму, что и входные данные.

    Raises:
        NormalizationError: Если операция нормализации не сработала из-за ошибки.

    Example:
        >> normalized = normalize_audio(audio_data, 44100)

    Note:
        Эта функция использует пиковую нормализацию.
    """
    logger.info("Применение нормализации громкости...")

    try:
        # Конвертация массива NumPy в AudioSegment с использованием временного WAV-буффера
        with io.BytesIO() as temp_buffer:
            sf.write(temp_buffer, data, sr, format="wav")
            temp_buffer.seek(0)

            # Загрузка аудио как AudioSegment и применение нормализации
            audio = AudioSegment.from_wav(temp_buffer)
            processed_audio = effects.normalize(audio)
            # processed_audio = effects.compress_dynamic_range(audio, threshold=-40.0, ratio=8.0)
            # Конвертация обратно в NumPy массив
            output_buffer = io.BytesIO()
            processed_audio.export(output_buffer, format="wav")
            output_buffer.seek(0)

            normalized, _ = sf.read(output_buffer)
            logger.info("Нормализация завершена")
        return normalized

    except Exception as ex:
        logger.error(f"Ошибка при нормализации громкости: {str(ex)}")
        raise app_exceptions.NormalizationError(f"Ошибка при нормализации громкости: {str(ex)}")


def process_audio_pipeline(input_path: str, output_path: str):
    """
    Функция с полным пайплайном обработки звука.
    Загрузка -> Удаление шума -> Фильтрация -> Нормализация -> Сохранение в выходной файл.

    Аудиофайл обрабатывается с помощью пайплайна:
    1. Загрузка и проверка аудиофайла.
    2. Удаление шума с помощью NSNet2Enhancer.
    3. Применение полосового фильтра Баттерворта для формирования диапазона частот.
    4. Нормализация громкости по пику.
    5. Сохранение обработанного звука в выходной файл.

    Args:
        input_path : str
            Путь к входному аудиофайлу (WAV или MP3).
        output_path : str
            Путь к обработанному аудиофайлу. Расширение определяется исходя из входного файла.

    Raises:
        AudioLoadError: Если не удается загрузить входной файл.
        AudioFormatError: Если неверный формат входного файла.
        AudioSaveError: Если выходной файл не может быть сохранен.
        NSnet2Enhancer: Если не удается выполнить удаление шума.
        FilterError: Если не удается применить фильтр.
        NormalizationError: Если не удается применить нормализацию.

    Example:
        >> process_audio_pipeline('noisy_input.wav', 'clean_output.wav')
        >> process_audio_pipeline('input.mp3', 'processed.mp3')

    Note:
        На всех этапах сборки используются параметры по умолчанию из config.py.
    """
    logger.info(f"Входной файл: {input_path} -> Выходной файл: {output_path}")
    logger.info("Запуск пайплайна обработки аудио.")

    # Шаг 1. Загрузка
    data, sr = load_audio(input_path)

    # Шаг 2. Удаление шума
    cleaned_data, sr = remove_noise(data, sr)

    # Шаг 3. Фильтрация
    filtered_data = apply_bandpass_filter(cleaned_data, sr)

    # Шаг 4. Нормализация
    normalized_data = normalize_audio(filtered_data, sr)

    # Шаг 5. Сохранение аудио
    save_audio(normalized_data, sr, output_path)

    logger.info("Пайплайн обработки завершен.")
