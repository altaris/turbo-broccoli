"""
numpy (de)serialization utilities.

Todo:
    Handle numpy's `generic` type (which supersedes the `number` type).
"""
__docformat__ = "google"

import pickle
from typing import Any, Callable, Tuple
from uuid import uuid4

import numpy as np
from safetensors import numpy as st

from turbo_broccoli.environment import get_artifact_path, get_max_nbytes
from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_dtype(dct: dict) -> np.dtype:
    """
    Converts a JSON document to a numpy dtype. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    raise_if_nodecode("numpy.dtype")
    DECODERS = {
        # 1: _json_to_dtype_v1,  # Use turbo_broccoli v3
        2: _json_to_dtype_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_dtype_v2(dct: dict) -> np.dtype:
    """
    Converts a JSON document to a numpy dtype object following the v1
    specification.
    """
    return np.lib.format.descr_to_dtype(dct["dtype"])


def _json_to_ndarray(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    raise_if_nodecode("numpy.ndarray")
    DECODERS = {
        # 1: _json_to_ndarray_v1,  # Use turbo_broccoli v3
        # 2: _json_to_ndarray_v2,  # Use turbo_broccoli v3
        # 3: _json_to_ndarray_v3,  # Use turbo_broccoli v3
        4: _json_to_ndarray_v4,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_ndarray_v4(dct: dict) -> np.ndarray:
    """
    Converts a JSON document to a numpy array following the v1 specification.
    """
    if "data" in dct:
        return st.load(dct["data"])["data"]
    return st.load_file(get_artifact_path() / dct["id"])["data"]


def _json_to_number(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy number. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    raise_if_nodecode("numpy.number")
    DECODERS = {
        # 1: _json_to_number_v1,  # Use turbo_broccoli v3
        # 2: _json_to_number_v2,  # Use turbo_broccoli v3
        3: _json_to_number_v3,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_number_v3(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy number following the v3 specification.
    """
    return np.frombuffer(dct["value"], dtype=dct["dtype"])[0]


def _json_to_random_state(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy random state. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__numpy__`
    should not be present.
    """
    raise_if_nodecode("numpy.random_state")
    DECODERS = {
        # 1: _json_to_random_state_v1,
        2: _json_to_random_state_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_random_state_v2(dct: dict) -> np.number:
    """
    Converts a JSON document to a numpy random state following the v2
    specification.
    """
    with open(get_artifact_path() / dct["data"], mode="rb") as fp:
        return pickle.load(fp)


def _dtype_to_json(d: np.dtype) -> dict:
    """Serializes a `numpy` array."""
    return {
        "__type__": "numpy.dtype",
        "__version__": 2,
        "dtype": np.lib.format.dtype_to_descr(d),
    }


def _ndarray_to_json(arr: np.ndarray) -> dict:
    """Serializes a `numpy` array."""
    if arr.nbytes <= get_max_nbytes():
        return {
            "__type__": "numpy.ndarray",
            "__version__": 4,
            "data": st.save({"data": arr}),
        }
    name = str(uuid4())
    st.save_file({"data": arr}, get_artifact_path() / name)
    return {
        "__type__": "numpy.ndarray",
        "__version__": 4,
        "id": name,
    }


def _number_to_json(num: np.number) -> dict:
    """Serializes a `numpy` number."""

    return {
        "__type__": "numpy.number",
        "__version__": 3,
        "value": bytes(np.array(num).data),
        "dtype": num.dtype,
    }


def _random_state_to_json(obj: np.random.RandomState) -> dict:
    """Pickles a numpy random state"""
    name, protocol = str(uuid4()), pickle.HIGHEST_PROTOCOL
    with open(get_artifact_path() / name, mode="wb") as fp:
        pickle.dump(obj, fp, protocol=protocol)
    return {
        "__type__": "numpy.random_state",
        "__version__": 2,
        "data": name,
        "protocol": protocol,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a numpy object. See `to_json` for the
    specification `dct` is expected to follow.
    """
    raise_if_nodecode("numpy")
    DECODERS = {
        "numpy.ndarray": _json_to_ndarray,
        "numpy.number": _json_to_number,
        "numpy.dtype": _json_to_dtype,
        "numpy.random_state": _json_to_random_state,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a `numpy` object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `numpy.ndarray`: An array is processed differently depending on its size
      and on the `TB_MAX_NBYTES` environment variable. If the array is
      small, i.e. `arr.nbytes <= TB_MAX_NBYTES`, then it is directly
      stored in the resulting JSON document as

            {
                "__type__": "numpy.ndarray",
                "__version__": 4,
                "data": <ASCII encoded byte string>,
            }


      On the other hand, the array is too large (i.e. the number of bytes
      exceeds `TB_MAX_NBYTES` or the value set by
      `turbo_broccoli.environment.set_max_nbytes`), then the content of `arr`
      is stored in an `.npy` file. Said file is saved to the path specified by
      the `TB_ARTIFACT_PATH` environment variable with a
      random UUID4 as filename. The resulting JSON document looks like

            {
                "__type__": "numpy.ndarray",
                "__version__": 4,
                "id": <UUID4 str>,
            }

      By default, `TB_MAX_NBYTES` is `8000` bytes, which should be enough
      to store an array of 1000 `float64`s, and `TB_ARTIFACT_PATH` is `./`.
      `TB_ARTIFACT_PATH` must point to an existing directory.

    - `numpy.number`:

            {
                "__type__": "numpy.number",
                "__version__": 3,
                "value": <float>,
                "dtype": {...},
            }

        where the `dtype` document follows the specification below.

    - `numpy.dtype`:

            {
                "__type__": "numpy.dtype",
                "__version__": 2,
                "dtype": <dtype_to_descr string>,
            }

    - `numpy.random.RandomState`:

            {
                "__type__": "numpy.random_state",
                "__version__": 2,
                "dtype": <uuid4>,
                "protocol": <int>
            }

      where the UUID4 points to a pickle file artefact, and the protocol is the
      pickle protocol.

    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (np.ndarray, _ndarray_to_json),
        (np.number, _number_to_json),
        (np.dtype, _dtype_to_json),
        (np.random.RandomState, _random_state_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
