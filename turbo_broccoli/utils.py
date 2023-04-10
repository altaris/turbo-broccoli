"""Various utilities and internal methods"""

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

_WARNED_ABOUT_SAFETENSORS = False


class DeserializationError(Exception):
    """Raised whenever something went wrong during deserialization"""


class SerializationError(Exception):
    """Raised whenever something went wrong during serialization"""


class TypeNotSupported(Exception):
    """
    `to_json` will raise that if they are fed types they cannot manage. This is
    fine, the dispatch in
    `turbo_broccoli.turbo_broccoli.TurboBroccoliEncoder.default` catches these
    and moves on to the next registered `to_json` method.
    """


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
