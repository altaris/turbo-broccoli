"""Various utilities and internal methods"""

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

_WARNED_ABOUT_SAFETENSORS = False


def warn_about_safetensors():
    """
    If safetensors is not installed, logs a warning message. This method may be
    called multiple times, but the message will only be logged once.
    """
    global _WARNED_ABOUT_SAFETENSORS  # pylint: disable=global-statement
    if not _WARNED_ABOUT_SAFETENSORS:
        logging.warning(
            "Serialization of numpy arrays and Tensorflow tensors without "
            "safetensors is deprecated. Consider installing safetensors using "
            "'pip install safetensors'."
        )
        _WARNED_ABOUT_SAFETENSORS = True
