"""Tensorflow (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple
from uuid import uuid4

import tensorflow as tf

try:
    from safetensors import tensorflow as st

    HAS_SAFETENSORS = True
except ModuleNotFoundError:
    HAS_SAFETENSORS = False
    from turbo_broccoli.utils import warn_about_safetensors

    warn_about_safetensors()

from turbo_broccoli.environment import (
    get_artifact_path,
    get_max_nbytes,
    is_nodecode,
)


def _json_to_sparse_tensor(dct: dict) -> tf.Tensor:
    """Converts a JSON document to a tensorflow tensor."""
    DECODERS = {
        1: _json_to_sparse_tensor_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_sparse_tensor_v1(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v1 specification to a tensorflow
    sparse tensor.
    """
    return tf.SparseTensor(
        dense_shape=dct["shape"],
        indices=dct["indices"],
        values=dct["values"],
    )


def _json_to_tensor(dct: dict) -> tf.Tensor:
    """Converts a JSON document to a tensorflow tensor."""
    DECODERS = {
        1: _json_to_tensor_v1,
        2: _json_to_tensor_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_tensor_v1(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v1 specification to a tensorflow
    tensor.
    """
    return tf.constant(dct["numpy"], dtype=dct["dtype"])


def _json_to_tensor_v2(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v1 specification to a tensorflow
    tensor.
    """
    if not HAS_SAFETENSORS:
        raise RuntimeError(
            "A v2 tensorflow tensor document cannot be deserialized without "
            "safetensors. Install safetensors using "
            "'pip install safetensors'"
        )
    if "data" in dct:
        return st.load(dct["data"])["data"]
    return st.load_file(get_artifact_path() / dct["id"])["data"]


def _json_to_variable(dct: dict) -> tf.Variable:
    """Converts a JSON document to a tensorflow variable."""
    DECODERS = {
        1: _json_to_variable_v1,
        2: _json_to_variable_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_variable_v1(dct: dict) -> tf.Variable:
    """
    Converts a JSON document following the v1 specification to a tensorflow
    variable.
    """
    return tf.Variable(
        dtype=dct["dtype"],
        initial_value=dct["numpy"],
        name=dct["name"],
        trainable=dct["trainable"],
    )


def _json_to_variable_v2(dct: dict) -> tf.Variable:
    """
    Converts a JSON document following the v2 specification to a tensorflow
    variable.
    """
    return tf.Variable(
        initial_value=dct["value"],
        name=dct["name"],
        trainable=dct["trainable"],
    )


def _ragged_tensor_to_json(tens: tf.Tensor) -> dict:
    """Serializes a general tensor"""
    raise NotImplementedError(
        "Serialization of ragged tensors is not supported"
    )


def _sparse_tensor_to_json(tens: tf.SparseTensor) -> dict:
    """Serializes a sparse tensor"""
    return {
        "__type__": "sparse_tensor",
        "__version__": 1,
        "indices": tens.indices,
        "shape": list(tens.dense_shape),
        "values": tens.values,
    }


def _tensor_to_json(tens: tf.Tensor) -> dict:
    """Serializes a general tensor"""
    return (
        _tensor_to_json_v2(tens)
        if HAS_SAFETENSORS
        else _tensor_to_json_v1(tens)
    )


def _tensor_to_json_v1(tens: tf.Tensor) -> dict:
    """
    Serializes a general tensor following the v1 format, (which doesn't use
    `safetensors`)
    """
    return {
        "__type__": "tensor",
        "__version__": 1,
        "dtype": tens.dtype.name,
        "numpy": tens.numpy(),
    }


def _tensor_to_json_v2(tens: tf.Tensor) -> dict:
    """
    Serializes a general tensor following the v2 format, (which does use
    `safetensors`)
    """
    if tens.numpy().nbytes <= get_max_nbytes():
        return {
            "__type__": "tensor",
            "__version__": 2,
            "data": st.save({"data": tens}),
        }
    name = str(uuid4())
    st.save_file({"data": tens}, get_artifact_path() / name)
    return {
        "__type__": "tensor",
        "__version__": 2,
        "id": name,
    }


def _variable_to_json(var: tf.Variable) -> dict:
    """Serializes a tensorflow variable"""
    return {
        "__type__": "variable",
        "__version__": 2,
        "name": var.name,
        "value": var.value(),
        "trainable": var.trainable,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a tensorflow object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__tensorflow__`.
    """
    DECODERS = {
        "sparse_tensor": _json_to_sparse_tensor,
        "tensor": _json_to_tensor,
        "variable": _json_to_variable,
    }
    try:
        type_name = dct["__tensorflow__"]["__type__"]
        if is_nodecode("tensorflow." + type_name):
            return None
        return DECODERS[type_name](dct["__tensorflow__"])
    except KeyError as exc:
        raise TypeError("Not a valid tensorflow document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a tensorflow object into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__tensorflow__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.

    * `tf.RaggedTensor`: Not supported.

    * `tf.SparseTensor`:

            {
                "__type__": "sparse_tensor",
                "__version__": 1,
                "indices": {...},
                "values": {...},
                "shape": {...},
            }

      where the first two `{...}` placeholders result in the serialization of
      `tf.Tensor` (see below).

    * other `tf.Tensor` subtypes:

            {
                "__type__": "tensor",
                "__version__": 1,
                "dtype": <str>,
                "numpy": {...},
            }

      or, if the `safetensors` package is available:

            {
                "__type__": "tensor",
                "__version__": 2,
                "dtype": <str>,
                "data": {...},
            }

      On the other hand, if the `safetensors` package is available, and if the
      tensor is too large (i.e. the number of bytes exceeds `TB_MAX_NBYTES` or
      the value set by `turbo_broccoli.environment.set_max_nbytes`), then the
      content of the tensor is stored in a binary artefact´. Said file is saved
      to the path specified by the `TB_ARTIFACT_PATH` environment variable with
      a random UUID4 as filename. The resulting JSON document looks like

            {
                "__numpy__": {
                    "__type__": "tensor",
                    "__version__": 2,
                    "id": <UUID4 str>,
                }
            }

      By default, `TB_MAX_NBYTES` is `8000` bytes, which should be enough
      to store an array of 1000 `float64`s, and `TB_ARTIFACT_PATH` is `./`.
      `TB_ARTIFACT_PATH` must point to an existing directory.

    * `tf.Variable`:

            {
                "__type__": "tensor",
                "__version__": 2,
                "name": <str>,
                "value": {...},
                "trainable": <bool>,
            }

      where `{...}` is the document produced by serializing the value tensor of
      the variable.

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (tf.RaggedTensor, _ragged_tensor_to_json),
        (tf.SparseTensor, _sparse_tensor_to_json),
        (tf.Tensor, _tensor_to_json),
        (tf.Variable, _variable_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__tensorflow__": f(obj)}
    raise TypeError("Not a supported tensorflow type")
