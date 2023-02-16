# pylint: disable=global-variable-not-assigned
# pylint: disable=missing-function-docstring
"""
Environment variable and settings management. See the
[README](https://cedric.hothanh.fr/turbo-broccoli/turbo_broccoli.html#environment-variables)
for information about the supported environment variables.
"""
__docformat__ = "google"

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

_DATACLASSES_TYPES: Dict[str, type] = {}
_PYTORCH_MODULE_TYPES: Dict[str, type] = {}

# The initial values are the defaults
_ENVIRONMENT: Dict[str, Any] = {
    "TB_ARTIFACT_PATH": Path("./"),
    "TB_KERAS_FORMAT": "tf",
    "TB_MAX_NBYTES": 8_000,
    "TB_NODECODE": [],
    "TB_PANDAS_FORMAT": "h5",
    "TB_SHARED_KEY": None,
}


def _init():
    """
    Reads the environment and sets the
    `turbo_broccoli.environment._ENVIRONMENT` accordingly.
    """
    if "TB_NUMPY_PATH" in os.environ:
        logging.warning(
            "The use of the 'TB_NUMPY_PATH' environment variable is "
            "deprecated. Consider using 'TB_ARTIFACT_PATH' instead."
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
            "Invalid value for environment variable TB_KERAS_FORMAT: "
            f"'{os.environ['TB_KERAS_FORMAT']}'. Expected 'h5', 'json', or "
            f"'tf'. Defaulting to '{_ENVIRONMENT['TB_KERAS_FORMAT']}'."
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
            "Invalid value for environment variable TB_PANDAS_FORMAT: "
            f"'{os.environ['TB_PANDAS_FORMAT']}'. Expected 'csv', 'excel', "
            "'feather', 'h5', 'hdf', 'pickle', 'stata', or 'xml'. Defaulting "
            f"to '{_ENVIRONMENT['TB_PANDAS_FORMAT']}'."
        )

    if "TB_NODECODE" in os.environ:
        set_nodecode(os.environ["TB_NODECODE"])

    if "TB_NUMPY_MAX_NBYTES" in os.environ:
        logging.warning(
            "The use of the 'TB_NUMPY_MAX_NBYTES' environment variable is "
            "deprecated. Consider using 'TB_MAX_NBYTES' instead."
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

    if "TB_SHARED_KEY" in os.environ:
        set_shared_key(os.environ["TB_SHARED_KEY"])


def get_artifact_path() -> Path:
    return _ENVIRONMENT["TB_ARTIFACT_PATH"]


def get_keras_format() -> str:
    return _ENVIRONMENT["TB_KERAS_FORMAT"]


def get_pandas_format() -> str:
    return _ENVIRONMENT["TB_PANDAS_FORMAT"]


def get_max_nbytes() -> int:
    return _ENVIRONMENT["TB_MAX_NBYTES"]


def get_registered_dataclass_type(name: str) -> type:
    return _DATACLASSES_TYPES[name]


def get_registered_pytorch_module_type(name: str) -> type:
    return _PYTORCH_MODULE_TYPES[name]


def get_shared_key() -> Optional[bytes]:
    return _ENVIRONMENT["TB_SHARED_KEY"]


def is_nodecode(type_name: str) -> bool:
    return type_name in _ENVIRONMENT["TB_NODECODE"]


def register_dataclass_type(cls: type):
    """
    Registers a dataclass for dataclass deserialization. Registered types may
    be overwritten.
    """
    _DATACLASSES_TYPES[cls.__name__] = cls


def register_pytorch_module_type(cls: type):
    """
    Registers a `torch.nn.Module` type for module deserialization. Registered
    types may be overwritten.
    """
    _PYTORCH_MODULE_TYPES[cls.__name__] = cls


def set_artifact_path(path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    if not (path.exists() and path.is_dir()):
        raise RuntimeError(
            f"Path {str(path)} does not point to an existing directory"
        )
    _ENVIRONMENT["TB_ARTIFACT_PATH"] = path


def set_keras_format(fmt: str):
    """Valid format are `h5`, `json`, and `tf`."""
    fmt = fmt.lower()
    KERAS_FORMATS = ["h5", "json", "tf"]
    if fmt not in KERAS_FORMATS:
        raise ValueError(
            f"Invalid value for environment variable TB_KERAS_FORMAT: {fmt}. "
            "Valid formats are 'h5', 'json', and 'tf' (the default)."
        )
    _ENVIRONMENT["TB_KERAS_FORMAT"] = fmt


def set_max_nbytes(nbytes: int):
    if nbytes <= 0:
        raise ValueError("numpy's max nbytes must be > 0")
    _ENVIRONMENT["TB_MAX_NBYTES"] = nbytes


def set_nodecode(types: Union[str, List[str]]):
    _ENVIRONMENT["TB_NODECODE"] = (
        types.split(",") if isinstance(types, str) else types
    )


def set_pandas_format(fmt: str):
    """
    Valid formats are `csv`, `excel`, `feather`, `h5`, `hdf`, `html`, `pickle`,
    `sql`, `stata`, and `xml`.

    TODO: Write all the unit tests.
    """
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
            f"Invalid value for environment variable TB_PANDAS_FORMAT: {fmt}. "
            "Valid formats are `csv`, `excel`, `feather`, `h5` (the default), "
            "`hdf`, `html`, `pickle`, `sql`, `stata`, and `xml`."
        )
    _ENVIRONMENT["TB_PANDAS_FORMAT"] = fmt


def set_shared_key(key: Optional[Union[str, bytes]]):
    """If the provided key is a string, it will be encoded in `utf-8`."""
    if isinstance(key, str):
        key = key.encode("utf-8")
    _ENVIRONMENT["TB_SHARED_KEY"] = key


_init()
