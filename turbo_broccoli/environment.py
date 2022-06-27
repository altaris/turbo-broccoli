# pylint: disable=missing-function-docstring
"""
Environment variable and settings management. See the README for information
about the supported environment variables.
"""
__docformat__ = "google"

import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

_DATACLASSES_TYPES: Dict[str, type] = {}

# The initial values are the defaults
_ENVIRONMENT: Dict[str, Any] = {
    "TB_ARTIFACT_PATH": Path("./"),
    "TB_KERAS_FORMAT": "tf",
    "TB_PANDAS_FORMAT": "h5",
    "TB_MAX_NBYTES": 8_000,
}


def _init():
    """
    Reads the environment and sets the
    `turbo_broccoli.environment._ENVIRONMENT` accordingly.
    """
    if "TB_NUMPY_PATH" in os.environ:
        logging.warning(
            "The use of the TB_NUMPY_PATH environment variable is deprecated. "
            "Consider using TB_ARTIFACT_PATH instead"
        )
        set_artifact_path(Path(os.environ["TB_NUMPY_PATH"]))
    else:
        set_artifact_path(
            os.environ.get(
                "TB_ARTIFACT_PATH",
                _ENVIRONMENT["TB_ARTIFACT_PATH"],
            )
        )

    try:
        set_keras_format(
            os.environ.get(
                "TB_KERAS_FORMAT",
                _ENVIRONMENT["TB_KERAS_FORMAT"],
            )
        )
    except ValueError:
        logging.warning(
            "Invalid value for environment variable TB_KERAS_FORMAT: '%s'. "
            "Expected 'h5', 'json', or 'tf'. Defaulting to '%s'",
            os.environ["TB_KERAS_FORMAT"],
            _ENVIRONMENT["TB_KERAS_FORMAT"],
        )

    try:
        set_pandas_format(
            os.environ.get(
                "TB_PANDAS_FORMAT",
                _ENVIRONMENT["TB_PANDAS_FORMAT"],
            )
        )
    except ValueError:
        logging.warning(
            "Invalid value for environment variable TB_PANDAS_FORMAT: '%s'. "
            "Expected 'csv', 'excel', 'feather', 'h5', 'hdf', 'pickle', "
            "'stata', or 'xml'. Defaulting to '%s'",
            os.environ["TB_PANDAS_FORMAT"],
            _ENVIRONMENT["TB_PANDAS_FORMAT"],
        )

    if "TB_NUMPY_MAX_NBYTES" in os.environ:
        logging.warning(
            "The use of the TB_NUMPY_MAX_NBYTES environment variable is "
            "deprecated. Consider using TB_MAX_NBYTES instead"
        )
        set_max_nbytes(int(os.environ["TB_NUMPY_MAX_NBYTES"]))
    else:
        set_max_nbytes(
            int(
                os.environ.get(
                    "TB_MAX_NBYTES",
                    _ENVIRONMENT["TB_MAX_NBYTES"],
                )
            )
        )


def get_artifact_path() -> Path:
    return _ENVIRONMENT["TB_ARTIFACT_PATH"]


def get_keras_format() -> str:
    return _ENVIRONMENT["TB_KERAS_FORMAT"]


def get_pandas_format() -> str:
    return _ENVIRONMENT["TB_PANDAS_FORMAT"]


def get_max_nbytes() -> int:
    return _ENVIRONMENT["TB_MAX_NBYTES"]


def get_registered_dataclass(name: str) -> type:
    return _DATACLASSES_TYPES[name]


def register_dataclass(name: str, cls: type):
    """
    Registers a dataclass for dataclass deserialization. Registered types may
    be overwritten.
    """
    _DATACLASSES_TYPES[name] = cls


def set_artifact_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    if not (path.exists() and path.is_dir()):
        raise RuntimeError(
            f"Path {str(path)} does not point to an existing directory"
        )
    _ENVIRONMENT["TB_ARTIFACT_PATH"] = path


def set_keras_format(fmt: str):
    fmt = fmt.lower()
    KERAS_FORMATS = ["h5", "json", "tf"]
    if fmt not in KERAS_FORMATS:
        raise ValueError(
            f"Invalid value for environment variable TB_KERAS_FORMAT: {fmt}."
        )
    _ENVIRONMENT["TB_KERAS_FORMAT"] = fmt


def set_pandas_format(fmt: str):
    fmt = fmt.lower()
    PANDAS_FORMATS = [
        "csv",
        "excel",
        "feather",
        "h5",
        "hdf",
        "html",
        "pickle",
        "sql",
        "stata",
        "xml",
    ]
    if fmt not in PANDAS_FORMATS:
        raise ValueError(
            f"Invalid value for environment variable TB_PANDAS_FORMAT: {fmt}."
        )
    _ENVIRONMENT["TB_PANDAS_FORMAT"] = fmt


def set_max_nbytes(nbytes: int):
    if nbytes <= 0:
        raise ValueError("numpy's max nbytes must be > 0")
    _ENVIRONMENT["TB_MAX_NBYTES"] = nbytes


_init()
