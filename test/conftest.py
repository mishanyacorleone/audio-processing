import pytest
from pydub.generators import Sine
import numpy as np
from unittest.mock import Mock
from fastapi import UploadFile


def generate_audio(file_path, format: str, duration_ms: int, channels: int = 1):
    tone = Sine(440).to_audio_segment(duration=duration_ms)
    audio = tone.set_channels(channels)
    audio.export(file_path, format=format)
    return str(file_path)


# Моно
@pytest.fixture
def mono_wav(tmp_path):
    return generate_audio(tmp_path / "mono.wav", format="wav", duration_ms=100, channels=1)


@pytest.fixture
def mono_mp3(tmp_path):
    return generate_audio(tmp_path / "mono.mp3", format="mp3", duration_ms=100, channels=1)


# Стерео
@pytest.fixture
def stereo_wav(tmp_path):
    return generate_audio(tmp_path / "stereo.wav", format="wav", duration_ms=100, channels=2)


@pytest.fixture
def stereo_mp3(tmp_path):
    return generate_audio(tmp_path / "stereo.mp3", format="mp3", duration_ms=100, channels=2)


# Пустой файл (0 ms) (возможно нужно ещё добавить на 2 канала)
@pytest.fixture
def empty_wav(tmp_path):
    return generate_audio(tmp_path / "empty.wav", format="wav", duration_ms=0, channels=1)


@pytest.fixture
def empty_mp3(tmp_path):
    return generate_audio(tmp_path / "empty.mp3", format="mp3", duration_ms=0, channels=1)


# Сломанный файл
@pytest.fixture
def broken_wav(tmp_path):
    file_path = tmp_path / "broken.wav"
    file_path.write_bytes(b"\x00\xFF\x00")
    return str(file_path)


@pytest.fixture
def broken_mp3(tmp_path):
    file_path = tmp_path / "broken.mp3"
    file_path.write_bytes(b"\x00\xFF\x00")
    return str(file_path)


# Не аудиофайл
@pytest.fixture
def fake_wav(tmp_path):
    file_path = tmp_path / "fake.wav"
    file_path.write_text("Не аудио")
    return str(file_path)


@pytest.fixture
def fake_mp3(tmp_path):
    file_path = tmp_path / "fake.mp3"
    file_path.write_text("Не аудио")
    return str(file_path)


# Фикстуры для тестирования обработки звука
@pytest.fixture
def mono_signal():
    sample_rate = 16000
    duration_sec = 1
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    signal = 0.5 * np.sin(2 * np.pi * 220 * t)
    noise = 0.1 * np.random.randn(len(t))
    data = signal + noise
    return data, sample_rate


@pytest.fixture
def stereo_signal():
    sample_rate = 16000
    duration_sec = 1
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    noise = 0.1 * np.random.randn(len(t))
    left = 0.5 * np.sin(2 * np.pi * 220 * t) + noise
    right = 0.5 * np.cos(2 * np.pi * 220 * t) + noise
    stereo = np.stack([left, right], axis=1)
    return stereo, sample_rate


@pytest.fixture
def silent_audio():
    sample_rate = 16000
    duration_sec = 1
    data = np.random.normal(0, 1e-5, size=int(sample_rate * duration_sec))
    return data, sample_rate


@pytest.fixture
def zero_audio():
    sample_rate = 16000
    duration_sec = 1
    data = np.zeros(shape=int(sample_rate * duration_sec))
    return data, sample_rate


@pytest.fixture
def nan_audio():
    sample_rate = 16000
    duration_sec = 1
    data = np.full(int(sample_rate * duration_sec), np.nan, dtype=np.float32)
    return data, sample_rate


@pytest.fixture
def filter_values():
    values = [
        {'lowcut': 300, 'highcut': 3400, 'order': 6, 'is_valid': True},
        {'lowcut': -1, 'highcut': 3400, 'order': 6, 'is_valid': False},
        {'lowcut': 300, 'highcut': -1, 'order': 6, 'is_valid': False},
        {'lowcut': 3400, 'highcut': 300, 'order': 6, 'is_valid': False},
        {'lowcut': 8001, 'highcut': 3400, 'order': 6, 'is_valid': False},
        {'lowcut': 300, 'highcut': 8001, 'order': 6, 'is_valid': False},
        {'lowcut': 300, 'highcut': 3400, 'order': -1, 'is_valid': False},
        {'lowcut': 300, 'highcut': 3400, 'order': 1000, 'is_valid': False},
        {'lowcut': 300, 'highcut': 3400, 'order': 16001, 'is_valid': False}
    ]
    return values


# Тестирование через Mock функции validate_uploaded_file

@pytest.fixture
def wav_content_type():
    file = Mock(spec=UploadFile)
    file.filename = "test_content_type.wav"
    file.content_type = "audio/wav"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def mp3_content_type():
    file = Mock(spec=UploadFile)
    file.filename = "test_content_type.mp3"
    file.content_type = "audio/mp3"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def mpeg_content_type():
    file = Mock(spec=UploadFile)
    file.filename = "test_content_type.mp3"
    file.content_type = "audio/mpeg"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def unsupported_content_type():
    file = Mock(spec=UploadFile)
    file.filename = "test_content_type.wav"
    file.content_type = "text/plain"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def wav_filename_lower():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.wav"
    file.content_type = "audio/wav"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def mp3_filename_lower():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.mp3"
    file.content_type = "audio/mp3"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def wav_filename_upper():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.WAV"
    file.content_type = "audio/wav"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def unsupported_format():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.txt"
    file.content_type = "audio/wav"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def valid_file_size():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.WAV"
    file.content_type = "audio/wav"
    file.size = 5 * 1024 * 1024
    return file


@pytest.fixture
def invalid_file_size():
    file = Mock(spec=UploadFile)
    file.filename = "test_filename.WAV"
    file.content_type = "audio/wav"
    file.size = 51 * 1024 * 1024
    return file

