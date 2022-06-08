# pylint: disable=missing-function-docstring
"""Environment variable and settings management."""
__docformat__ = "google"

import os
from pathlib import Path
from typing import Any, Dict
import logging

# The initial values are the defaults
_ENVIRONMENT: Dict[str, Any] = {
    "TB_ARTIFACT_PATH": Path("./"),
    "TB_NUMPY_MAX_NBYTES": 8_000,
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
        _ENVIRONMENT["TB_ARTIFACT_PATH"] = Path(os.environ["TB_NUMPY_PATH"])
    else:
        _ENVIRONMENT["TB_ARTIFACT_PATH"] = Path(
            os.environ.get(
                "TB_ARTIFACT_PATH",
                _ENVIRONMENT["TB_ARTIFACT_PATH"],
            )
        )
    _ENVIRONMENT["TB_NUMPY_MAX_NBYTES"] = int(
        os.environ.get(
            "TB_NUMPY_MAX_NBYTES",
            _ENVIRONMENT["TB_NUMPY_MAX_NBYTES"],
        )
    )


def get_artifact_path() -> Path:
    return _ENVIRONMENT["TB_ARTIFACT_PATH"]


def get_numpy_max_nbytes() -> int:
    return _ENVIRONMENT["TB_NUMPY_MAX_NBYTES"]


def set_artifact_path(path: Path):
    if path.exists() and path.is_dir():
        _ENVIRONMENT["TB_ARTIFACT_PATH"] = path
    raise RuntimeError(
        f"Path {str(path)} does not point to an existing directory"
    )


def set_numpy_max_nbytes(nbytes: int) -> Path:
    if nbytes > 0:
        _ENVIRONMENT["TB_NUMPY_MAX_NBYTES"] = nbytes
    raise ValueError("numpy's max nbytes must be > 0")


_init()
