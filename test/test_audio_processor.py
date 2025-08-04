import pytest
import numpy as np

from app.src.audio_processor import \
    load_audio, is_valid_audio, \
    remove_noise, \
    apply_bandpass_filter, normalize_audio

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
        ("empty_wav", True),
        ("empty_mp3", True)
    ]
)
def test_is_valid_audio(request, fixture_name, expectation):
    test_file = request.getfixturevalue(fixture_name)
    assert is_valid_audio(str(test_file)) is expectation


@pytest.mark.parametrize(
    "fixture_name, should_pass, expectation",
    [
        ("mono_wav", True, does_not_raise()),
        ("mono_mp3", True, does_not_raise()),
        ("stereo_wav", True, does_not_raise()),
        ("stereo_mp3", True, does_not_raise()),
        ("fake_wav", False, exceptions.AudioFormatError),
        ("fake_mp3", False, exceptions.AudioFormatError),
        ("broken_wav", False, exceptions.AudioFormatError),
        ("broken_mp3", False, exceptions.AudioFormatError),
        ("empty_wav", False, exceptions.AudioFormatError),
        ("empty_mp3", False, exceptions.AudioFormatError)
    ]
)
def test_load_audio(request, should_pass, fixture_name, expectation):
    test_file = request.getfixturevalue(fixture_name)
    if should_pass:
        data, sr = load_audio(str(test_file))
        assert isinstance(data, np.ndarray)
        assert isinstance(sr, int)
        assert data.size > 0
        assert sr > 0
    else:
        with pytest.raises(expectation):
            load_audio(str(test_file))


@pytest.mark.parametrize(
    "fixture_name, should_pass, expectation",
    [
        ("mono_signal", True, does_not_raise()),
        ("stereo_signal", True, does_not_raise()),
        ("silent_audio", True, does_not_raise()),
        ("zero_audio", False, exceptions.AudioFormatError),
        ("nan_audio", False, exceptions.AudioFormatError)
    ]
)
def test_remove_noise(request, should_pass, fixture_name, expectation):
    data, sr = request.getfixturevalue(fixture_name)
    print(data, sr)
    if should_pass:
        cleaned, sr = remove_noise(data, sr)
        assert isinstance(cleaned, np.ndarray)
        assert cleaned.shape == data.shape
        assert not np.any(np.isnan(cleaned))
    else:
        with pytest.raises(expectation):
            remove_noise(data, sr)


@pytest.mark.parametrize(
    "audio_fixture", ["mono_signal", "stereo_signal", "silent_audio"]
)
@pytest.mark.parametrize(
    "values", ["filter_values"]
)
def test_apply_bandpass_filter(request, audio_fixture, values):
    data, sr = request.getfixturevalue(audio_fixture)
    filter_values = request.getfixturevalue(values)
    for params in filter_values:
        lowcut = params['lowcut']
        highcut = params['highcut']
        order = params['order']
        is_valid = params['is_valid']
        if is_valid:
            filtered = apply_bandpass_filter(data, sr, lowcut, highcut, order)
            assert isinstance(filtered, np.ndarray)
            assert filtered.shape == data.shape
            assert not np.any(np.isnan(filtered))
            assert not np.any(np.isinf(filtered))
        else:
            with pytest.raises(exceptions.AudioParameterError):
                apply_bandpass_filter(data, sr, lowcut, highcut, order)


@pytest.mark.parametrize(
    "fixture_name",
    [
        "mono_signal",
        "stereo_signal",
        "silent_audio",
    ]
)
def test_normalize_audio(request, fixture_name):
    data, sr = request.getfixturevalue(fixture_name)
    cleaned = normalize_audio(data, sr)
    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == data.shape
    assert not np.any(np.isnan(cleaned))
    assert not np.any(np.isinf(cleaned))