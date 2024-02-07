"""
numpy (de)serialization utilities.

Todo:
    Handle numpy's `generic` type (which supersedes the `number` type).
"""

import pickle
from typing import Any, Callable, Tuple

import numpy as np
from safetensors import numpy as st

from turbo_broccoli.context import Context
from turbo_broccoli.utils import DeserializationError, TypeNotSupported


def _json_to_dtype(dct: dict, ctx: Context) -> np.dtype:
    DECODERS = {
        # 1: _json_to_dtype_v1,  # Use turbo_broccoli v3
        2: _json_to_dtype_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_dtype_v2(dct: dict, ctx: Context) -> np.dtype:
    return np.lib.format.descr_to_dtype(dct["dtype"])


def _json_to_ndarray(dct: dict, ctx: Context) -> np.ndarray:
    DECODERS = {
        # 1: _json_to_ndarray_v1,  # Use turbo_broccoli v3
        # 2: _json_to_ndarray_v2,  # Use turbo_broccoli v3
        # 3: _json_to_ndarray_v3,  # Use turbo_broccoli v3
        4: _json_to_ndarray_v4,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_ndarray_v4(dct: dict, ctx: Context) -> np.ndarray:
    if "data" in dct:
        return st.load(dct["data"])["data"]
    return st.load_file(ctx.artifact_path / (dct["id"] + ".tb"))["data"]


def _json_to_number(dct: dict, ctx: Context) -> np.number:
    DECODERS = {
        # 1: _json_to_number_v1,  # Use turbo_broccoli v3
        # 2: _json_to_number_v2,  # Use turbo_broccoli v3
        3: _json_to_number_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_number_v3(dct: dict, ctx: Context) -> np.number:
    return np.frombuffer(dct["value"], dtype=dct["dtype"])[0]


def _json_to_random_state(dct: dict, ctx: Context) -> np.number:
    DECODERS = {
        # 1: _json_to_random_state_v1,
        2: _json_to_random_state_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_random_state_v2(dct: dict, ctx: Context) -> np.number:
    with open(ctx.artifact_path / (dct["data"] + ".tb"), mode="rb") as fp:
        return pickle.load(fp)


def _dtype_to_json(d: np.dtype, ctx: Context) -> dict:
    return {
        "__type__": "numpy.dtype",
        "__version__": 2,
        "dtype": np.lib.format.dtype_to_descr(d),
    }


def _ndarray_to_json(arr: np.ndarray, ctx: Context) -> dict:
    if arr.nbytes <= ctx.min_artifact_size:
        return {
            "__type__": "numpy.ndarray",
            "__version__": 4,
            "data": st.save({"data": arr}),
        }
    path, name = ctx.new_artifact_path()
    st.save_file({"data": arr}, path)
    return {
        "__type__": "numpy.ndarray",
        "__version__": 4,
        "id": name,
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
    protocol = pickle.HIGHEST_PROTOCOL
    with open(path, mode="wb") as fp:
        pickle.dump(obj, fp, protocol=protocol)
    return {
        "__type__": "numpy.random_state",
        "__version__": 2,
        "data": name,
        "protocol": protocol,
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

            {
                "__type__": "numpy.ndarray",
                "__version__": 4,
                "data": <ASCII encoded byte string>,
            }


      On the other hand, the array is too large (see
      `turbo_broccoli.context.Context.min_artifact_size`), then the content of
      `arr` is stored in an `.npy` file and the resulting JSON document looks
      like

            {
                "__type__": "numpy.ndarray",
                "__version__": 4,
                "id": <UUID4 str>,
            }

      where `id` points to an artifact. By default, `TB_MAX_NBYTES` is `8000`
      bytes, which should be enough to store an array of 1000 `float64`s.

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
      [pickle
      protocol](`https://docs.python.org/3/library/pickle.html#data-stream-format`).
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
