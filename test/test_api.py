import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from fastapi import HTTPException
from app.main import app
from app.api import api
from app.src import exceptions
from contextlib import nullcontext as does_not_raise


@pytest.mark.parametrize(
    "fixture_name, expectation",
    [
        ("mono_wav", True),
        ("mono_mp3", True),
        ("stereo_wav", True),
        ("stereo_mp3", True),
        ("fake_wav", False),
        ("fake_mp3", False),
        ("broken_wav", False),
        ("broken_mp3", False),
        ("empty_wav", False),
        ("empty_mp3", False)
    ]
)
@pytest.mark.asyncio
async def test_process_audio(request, fixture_name, expectation):
    file_path = request.getfixturevalue(fixture_name)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/wav")}
            response = await ac.post(url="/api/process-audio/", files=files)

    if expectation:
        assert response.status_code == 200

    else:
        assert response.status_code == 400


@pytest.mark.parametrize(
    "exception, should_pass, status_code",
    [
        (exceptions.AudioFormatError(), True, 400),
        (exceptions.AudioParameterError(), True, 400),
        (exceptions.AudioProcessingError(), False, 500),
        (exceptions.AudioLoadError(), True, 500),
        (exceptions.AudioSaveError(), True, 500),
        (exceptions.NSNet2EnhancerError(), True, 500),
        (exceptions.NormalizationError(), True, 500),
        (exceptions.FilterError(), True, 500),
        (TypeError, False, 500),
        (ValueError, False, 500)
    ]
)
def test_map_audio_exception_to_http(exception, should_pass, status_code):
    http_ex = api.map_audio_exception_to_http(exception)
    if should_pass:
        assert type(http_ex) == HTTPException
        assert http_ex.status_code == status_code
    else:
        assert "Неизвестная ошибка обработки:" in http_ex.detail
        assert type(http_ex) == HTTPException
        assert http_ex.status_code == status_code


@pytest.mark.parametrize(
    "fixture_name, should_pass, expectation",
    [
        ("wav_content_type", True, does_not_raise()),
        ("mp3_content_type", True, does_not_raise()),
        ("mpeg_content_type", True, does_not_raise()),
        ("unsupported_content_type", False, HTTPException),
        ("wav_filename_lower", True, does_not_raise()),
        ("mp3_filename_lower", True, does_not_raise()),
        ("wav_filename_upper", True, does_not_raise()),
        ("unsupported_format", False, HTTPException),
        ("valid_file_size", True, does_not_raise()),
        ("invalid_file_size", False, HTTPException),
    ]
)
def test_validate_uploaded_file(request, fixture_name, should_pass, expectation):
    file = request.getfixturevalue(fixture_name)
    if should_pass:
        api.validate_uploaded_file(file)
    else:
        with pytest.raises(expectation):
            api.validate_uploaded_file(file)


@pytest.mark.parametrize(
    "filename, output",
    [
        ('test.wav', '.wav'),
        ('test.mp3', '.mp3'),
        ('test.MP3', '.mp3'),
        ('test.WAV', '.wav'),
    ]
)
def test_get_file_extension(filename, output):
    assert api.get_file_extension(filename) == output