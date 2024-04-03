"""
numpy (de)serialization utilities.

Todo:
    Handle numpy's `generic` type (which supersedes the `number` type).
"""

from typing import Any, Callable, Tuple

import joblib
import numpy as np
from safetensors import numpy as st

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_dtype(dct: dict, ctx: Context) -> np.dtype:
    DECODERS = {
        2: _json_to_dtype_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_dtype_v2(dct: dict, ctx: Context) -> np.dtype:
    return np.lib.format.descr_to_dtype(dct["dtype"])


def _json_to_ndarray(dct: dict, ctx: Context) -> np.ndarray:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        5: _json_to_ndarray_v5,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_ndarray_v5(dct: dict, ctx: Context) -> np.ndarray:
    return st.load(dct["data"])["data"]


def _json_to_number(dct: dict, ctx: Context) -> np.number:
    DECODERS = {
        3: _json_to_number_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_number_v3(dct: dict, ctx: Context) -> np.number:
    return np.frombuffer(dct["value"], dtype=dct["dtype"])[0]


def _json_to_random_state(dct: dict, ctx: Context) -> np.number:
    DECODERS = {
        3: _json_to_random_state_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_random_state_v3(dct: dict, ctx: Context) -> np.number:
    return joblib.load(ctx.id_to_artifact_path(dct["data"]))


def _dtype_to_json(d: np.dtype, ctx: Context) -> dict:
    return {
        "__type__": "numpy.dtype",
        "__version__": 2,
        "dtype": np.lib.format.dtype_to_descr(d),
    }


def _ndarray_to_json(arr: np.ndarray, ctx: Context) -> dict:
    return {
        "__type__": "numpy.ndarray",
        "__version__": 5,
        "data": st.save({"data": arr}),
    }


def _number_to_json(num: np.number, ctx: Context) -> dict:
    return {
        "__type__": "numpy.number",
        "__version__": 3,
        "value": bytes(np.array(num).data),
        "dtype": num.dtype,
    }


def _random_state_to_json(obj: np.random.RandomState, ctx: Context) -> dict:
    path, name = ctx.new_artifact_path()
    with path.open(mode="wb") as fp:
        joblib.dump(obj, fp)
    return {
        "__type__": "numpy.random_state",
        "__version__": 3,
        "data": name,
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    """
    Deserializes a dict into a numpy object. See `to_json` for the
    specification `dct` is expected to follow.
    """
    DECODERS = {
        "numpy.ndarray": _json_to_ndarray,
        "numpy.number": _json_to_number,
        "numpy.dtype": _json_to_dtype,
        "numpy.random_state": _json_to_random_state,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a `numpy` object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `numpy.ndarray`: An array is processed differently depending on its size
      and on the `TB_MAX_NBYTES` environment variable. If the array is
      small, i.e. `arr.nbytes <= TB_MAX_NBYTES`, then it is directly
      stored in the resulting JSON document as

        ```py
        {
            "__type__": "numpy.ndarray",
            "__version__": 5,
            "data": {
                "__type__": "bytes",
                ...
            }
        }
        ```

      see `turbo_broccoli.custom.bytes.to_json`.

    - `numpy.number`:

        ```py
        {
            "__type__": "numpy.number",
            "__version__": 3,
            "value": <float>,
            "dtype": {...},
        }
        ```

        where the `dtype` document follows the specification below.

    - `numpy.dtype`:

        ```py
        {
            "__type__": "numpy.dtype",
            "__version__": 2,
            "dtype": <dtype_to_descr string>,
        }
        ```

    - `numpy.random.RandomState`:

        ```py
        {
            "__type__": "numpy.random_state",
            "__version__": 3,
            "data": <uuid4>,
        }
        ```

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (np.ndarray, _ndarray_to_json),
        (np.number, _number_to_json),
        (np.dtype, _dtype_to_json),
        (np.random.RandomState, _random_state_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
