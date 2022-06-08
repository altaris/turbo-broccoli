"""
numpy (de)serialization utilities.

Todo:
    Handle numpy's `generic` type (which supersedes the `number` type).
"""
__docformat__ = "google"

import os
import logging
from pathlib import Path
from typing import Any, Callable, List, Tuple
from uuid import uuid4

import numpy as np


DEFAULT_TB_NUMPY_MAX_NBYTES = 8000


def _get_artifact_path() -> Path:
    """
    Reads the `TB_NUMPY_PATH` or `TB_ARTIFACT_PATH` environment variable, but logs a
    warning in the former case.
    """
    if "TB_NUMPY_PATH" in os.environ:
        logging.warning(
            "The use of the TB_NUMPY_PATH environment variable is deprecated. "
            "Consider using TB_ARTIFACT_PATH instead"
        )
        return Path(os.environ.get("TB_NUMPY_PATH", "./"))
    return Path(os.environ.get("TB_ARTIFACT_PATH", "./"))


def _json_to_ndarray(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_ndarray_v1,
        2: _json_to_ndarray_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_ndarray_v1(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array following the v1 specification.
    """
    return np.frombuffer(
        dct["data"],
        dtype=np.lib.format.descr_to_dtype(dct["dtype"]),
    ).reshape(dct["shape"])


def _json_to_ndarray_v2(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array following the v1 specification.
    """
    if "data" in dct:
        return _json_to_ndarray_v1(dct)
    path = _get_artifact_path()
    return np.load(path / (dct["id"] + ".npy"))


def _json_to_number(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy number. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_number_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_number_v1(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy number following the v1 specification.
    """
    return np.frombuffer(
        dct["value"],
        dtype=np.lib.format.descr_to_dtype(dct["dtype"]),
    )[0]


def _ndarray_to_json(arr: np.ndarray) -> dict:
    """Serializes a `numpy` array."""
    max_nbytes = int(
        os.environ.get("TB_NUMPY_MAX_NBYTES", DEFAULT_TB_NUMPY_MAX_NBYTES)
    )
    if arr.nbytes <= max_nbytes:
        return {
            "__type__": "ndarray",
            "__version__": 2,
            "data": bytes(arr.data),
            "dtype": np.lib.format.dtype_to_descr(arr.dtype),
            "shape": arr.shape,
        }
    path = _get_artifact_path()
    name = str(uuid4())
    np.save(path / name, arr)
    return {
        "__type__": "ndarray",
        "__version__": 2,
        "id": name,
    }


def _number_to_json(num: np.number) -> dict:
    """Serializes a `numpy` number."""

    return {
        "__type__": "number",
        "__version__": 1,
        "value": bytes(np.array(num).data),
        "dtype": np.lib.format.dtype_to_descr(num.dtype),
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a numpy object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__numpy__`.
    """
    DECODERS = {
        "ndarray": _json_to_ndarray,
        "number": _json_to_number,
    }
    try:
        return DECODERS[dct["__numpy__"]["__type__"]](dct["__numpy__"])
    except KeyError as exc:
        raise TypeError("Not a valid numpy document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a `numpy` object into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__numpy__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.

    * `numpy.ndarray`: An array is processed differently depending on its size
      and on the `TB_NUMPY_MAX_NBYTES` environment variable. If the array is
      small, i.e. `arr.nbytes <= TB_NUMPY_MAX_NBYTES`, then it is directly
      stored in the resulting JSON document as

            {
                "__numpy__": {
                    "__type__": "ndarray",
                    "__version__": 2,
                    "data": <ASCII encoded byte string>,
                    "dtype": <str>,
                    "shape": <int tuple>,
                }
            }

      On the other hand, if `arr.nbytes > TB_NUMPY_MAX_NBYTES`, then the
      content of `arr` is stored in an `.npy` file. Said file is saved to the
      path specified by the `TB_ARTIFACT_PATH` environment variable with a
      random UUID4 as filename. The resulting JSON document looks like

            {
                "__numpy__": {
                    "__type__": "ndarray",
                    "__version__": 2,
                    "id": <UUID4 str>,
                }
            }

      By default, `TB_NUMPY_MAX_NBYTES` is `8000` bytes, which should be enough
      to store an array of 1000 `float64`s, and `TB_ARTIFACT_PATH` is `./`.
      `TB_ARTIFACT_PATH` must point to an existing directory.

    * `numpy.number`:

            {
                "__numpy__": {
                    "__type__": "number",
                    "__version__": 1,
                    "value": <float>,
                    "dtype": <str>,
                }
            }

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (np.ndarray, _ndarray_to_json),
        (np.number, _number_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__numpy__": f(obj)}
    raise TypeError("Not a supported numpy type")
