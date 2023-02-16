"""Various utilities and internal methods"""

import logging

WARNED_ABOUT_SAFETENSORS = False


def warn_about_safetensors():
    """
    If safetensors is not installed, logs a warning message. This method may be
    called multiple times, but the message will only be logged once.
    """
    global WARNED_ABOUT_SAFETENSORS  # pylint: disable=global-statement
    if not WARNED_ABOUT_SAFETENSORS:
        logging.warning(
            "Serialization of numpy arrays and Tensorflow tensors without "
            "safetensors is deprecated. Consider installing safetensors using "
            "'pip install safetensors'"
        )
        WARNED_ABOUT_SAFETENSORS = True
