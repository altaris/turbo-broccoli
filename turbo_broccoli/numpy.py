"""
numpy (de)serialization utilities.

Todo:
    Handle numpy's `generic` type (which supersedes the `number` type).
"""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple
from base64 import b64decode, b64encode

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False

VERSION = 1
"""numpy document format version"""


def _json_to_ndarray(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_ndarray_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_ndarray_v1(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array following the v1 specification.
    """
    return np.frombuffer(
        b64decode(dct["value"]),
        dtype=np.lib.format.descr_to_dtype(dct["dtype"]),
    ).reshape(dct["shape"])


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
        b64decode(dct["value"]),
        dtype=np.lib.format.descr_to_dtype(dct["dtype"]),
    )[0]


def _ndarray_to_json(arr: np.ndarray) -> dict:
    """Serializes a `numpy` array."""
    return {
        "__type__": "ndarray",
        "__version__": VERSION,
        "data": b64encode(arr.data).decode("ASCII"),
        "dtype": np.lib.format.dtype_to_descr(arr.dtype),
        "shape": arr.shape,
    }


def _number_to_json(num: np.number) -> dict:
    """Serializes a `numpy` number."""

    return {
        "__type__": "number",
        "__version__": VERSION,
        "value": b64encode(np.array(num).data).decode("ASCII"),
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

    * `numpy.ndarray`:

            {
                "__numpy__": {
                    "__type__": "ndarray",
                    "__version__": <str>,
                    "data": <ASCII encoded byte string>,
                    "dtype": <str>,
                    "shape": <int tuple>,
                }
            }

    * `numpy.number`:

            {
                "__numpy__": {
                    "__type__": "number",
                    "__version__": <str>,
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
