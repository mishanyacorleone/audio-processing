# Audio Processing API

Сервис для обработки аудиофайлов с функциями удаления шумов, фильтрации по частотам и нормализации громкости.

## 📋 Описание
Audio Processing API - это REST API сервис, разработанный на FastAPI, который предоставляет возможности для обработки аудиофайлов в форматах WAV и MP3. Сервис выполняет следующие операции:
- **Удаление шумов** с использованием предобученной модели NSNet2Enhancer
- **Полосовая фильтрация** с помощью фильтра Баттерворта (300-3400 Гц)
- **Нормализация громкости** по пику

## 🏗️ Структура проекта
```
audio-processing/
├── app/
│   ├── api/
│   │   └── api.py              # Основной файл API
│   ├── config/
│   │   └── config.py           # Конфигурационные файлы
│   ├── models/                 # Модели данных (если используются)
│   ├── src/                    # Исходные скрипты приложения
│   │   ├── audio_processor.py  # Модуль обработки аудио
│   │   └── exceptions.py       # Пользовательские исключения
│   ├── logs/                   # Логи приложения
│   │   └── app.log            # Файл логов
│   └── main.py                # Основной файл приложения
├── data/
│   ├── input/                 # Временные входные файлы
│   └── output/                # Обработанные файлы
├── README.md
└── requirements.txt
```

## 🚀 Установка и запуск

### Требования

- Python 3.10
- pip

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск сервиса

```bash
# Из корневой директории проекта
python -m app.main

# Или с помощью uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Сервис будет доступен по адресу: `http://127.0.0.1:8000`

## 📖 API Документация

### Swagger UI

Интерактивная документация API доступна по адресу:
- **Swagger UI**: `http://127.0.0.1:8000/docs`

### Основные эндпоинты

#### POST `/api/process-audio/`

Загружает и обрабатывает аудиофайл.

**Параметры запроса:**
- `file` (form-data): Аудиофайл в формате WAV или MP3 (максимум 50 МБ)

**Пример ответа:**
```json
{
  "processed_file": "http://127.0.0.1:8000/output/processed_90b99262-1cbc-4640-8a31-fdfcc3e526c3.wav"
}
```

**Коды ответов:**
- `200` - Успешная обработка
- `400` - Ошибка валидации файла (неподдерживаемый формат, размер и т.д.)
- `500` - Внутренняя ошибка сервера

#### GET `/`

Информация о сервисе.

**Пример ответа:**
```json
{
  "message": "Audio Processing API. Документация: /docs"
}
```

## 🔧 Конфигурация

Основные параметры настройки находятся в `app/config/config.py`:

```python
# Параметры фильтрации
FILTER_LOW = 300.0      # Нижняя частота среза (Гц)
FILTER_HIGH = 3400.0    # Верхняя частота среза (Гц)
FILTER_ORDER = 6        # Порядок фильтра

# Ограничения
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 МБ
```

## 📝 Логирование

Логи записываются в:
- **Файл**: `app/logs/app.log`
- **Консоль**: stdout

Формат логов: `%(asctime)s - %(levelname)s - %(message)s`

## 🧪 Использование

### Пример с curl

```bash
curl -X POST "http://127.0.0.1:8000/api/process-audio/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_audio_file.wav;type=audio/wav"
```

### Пример с Python requests

```python
import requests

url = "http://127.0.0.1:8000/api/process-audio/"
file_path = "path/to/your/audio_file.wav"
with open(file_path, "rb") as f:
    files = {"file": ("audio_file.wav", f, "audio/wav")}
    response = requests.post(url, files=files)

    result = response.json()

print(f"Обработанный файл: {result['processed_file']}")
```

## 🔍 Этапы обработки

1. **Валидация файла** - проверка формата, размера и содержимого
2. **Загрузка аудио** - чтение аудиоданных с проверкой целостности
3. **Удаление шумов** - шумоподавление с использованием предобученной модели NSNet2Enhancer от Microsoft
4. **Фильтрация** - применение полосового фильтра Баттерворта (300-3400 Гц)
5. **Нормализация** - нормализация громкости по пику
6. **Сохранение** - сохранение результата в выходную директорию

## ⚠️ Обработка ошибок

Сервис обрабатывает следующие типы ошибок:

- **AudioFormatError** - неверный формат файла
- **AudioParameterError** - некорректные параметры обработки
- **AudioLoadError** - ошибки загрузки файла
- **AudioSaveError** - ошибки сохранения
- **NSNet2EnhancerError** - ошибки шумоподавления
- **FilterError** - ошибки фильтрации
- **NormalizationError** - ошибки нормализации

## 🐳 Docker

### Создание Dockerfile

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y gcc g++ ffmpeg pkg-config libsndfile1-dev libmagic-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /audio-processing

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/audio-processing

EXPOSE 8000

CMD ["python", "app/main.py"]
```

### Сборка и запуск контейнера

```bash
# Сборка образа
docker build -t audio-processing-api .

# Запуск контейнера
docker run --rm -p 8000:8000 audio-processing-api
```

## 🛠️ Разработка

### Тестирование

Для запуска тестов используйте pytest:

```bash
pytest tests/ -v --cov=app --cov-report=html
```

### Стиль кода

Код соответствует стандарту PEP8 и документирован в формате Google Style Guide.

## 📋 Ограничения

- Максимальный размер файла: 50 МБ
- Поддерживаемые форматы: WAV, MP3
- Сервис работает без доступа к интернету (после установки зависимостей)
- Все модели и библиотеки работают локально

## 📞 Поддержка

При возникновении вопросов или проблем:

1. Проверьте логи в `app/logs/app.log`
2. Убедитесь, что все зависимости установлены
3. Проверьте формат и размер загружаемого файла
4. Обратитесь к документации API по адресу `/docs`