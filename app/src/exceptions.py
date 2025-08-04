class AudioProcessingError(Exception):
    """Базовое исключение для ошибок обработки аудио"""
    pass


class AudioLoadError(AudioProcessingError):
    """Ошибка при загрузке аудиофайла"""
    pass


class AudioSaveError(AudioProcessingError):
    """Ошибка при сохранении аудиофайла"""
    pass


class AudioFormatError(AudioProcessingError):
    """Ошибка формата аудио"""
    pass


class AudioParameterError(AudioProcessingError):
    """Ошибка параметров обработки аудио"""
    pass


class NSNet2EnhancerError(AudioProcessingError):
    """Ошибка при удалении шума"""
    pass


class FilterError(AudioProcessingError):
    """Ошибка при применении фильтра"""
    pass


class NormalizationError(AudioProcessingError):
    """Ошибка при применении нормализации"""
    pass
