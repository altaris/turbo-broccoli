"""Tensorflow (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, Tuple
from uuid import uuid4

import tensorflow as tf

from safetensors import tensorflow as st

from turbo_broccoli.environment import (
    get_artifact_path,
    get_max_nbytes,
)
from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_sparse_tensor(dct: dict) -> tf.Tensor:
    """Converts a JSON document to a tensorflow tensor."""
    DECODERS = {
        # 1: _json_to_sparse_tensor_v1,  # Use turbo_broccoli v3
        2: _json_to_sparse_tensor_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_sparse_tensor_v2(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v2 specification to a tensorflow
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
        # 1: _json_to_tensor_v1,  # Use turbo_broccoli v3
        # 2: _json_to_tensor_v2,  # Use turbo_broccoli v3
        3: _json_to_tensor_v3,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_tensor_v3(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v2 specification to a tensorflow
    tensor.
    """
    if "data" in dct:
        return st.load(dct["data"])["data"]
    return st.load_file(get_artifact_path() / dct["id"])["data"]


def _json_to_variable(dct: dict) -> tf.Variable:
    """Converts a JSON document to a tensorflow variable."""
    DECODERS = {
        # 1: _json_to_variable_v1,  # Use turbo_broccoli v3
        # 2: _json_to_variable_v2,  # Use turbo_broccoli v3
        3: _json_to_variable_v3,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_variable_v3(dct: dict) -> tf.Variable:
    """
    Converts a JSON document following the v3 specification to a tensorflow
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
        "__type__": "tensorflow.sparse_tensor",
        "__version__": 2,
        "indices": tens.indices,
        "shape": list(tens.dense_shape),
        "values": tens.values,
    }


def _tensor_to_json(tens: tf.Tensor) -> dict:
    """
    Serializes a general tensor following the v3 format.
    """
    if tens.numpy().nbytes <= get_max_nbytes():
        return {
            "__type__": "tensorflow.tensor",
            "__version__": 3,
            "data": st.save({"data": tens}),
        }
    name = str(uuid4())
    st.save_file({"data": tens}, get_artifact_path() / name)
    return {
        "__type__": "tensorflow.tensor",
        "__version__": 3,
        "id": name,
    }


def _variable_to_json(var: tf.Variable) -> dict:
    """Serializes a tensorflow variable"""
    return {
        "__type__": "tensorflow.variable",
        "__version__": 3,
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
    raise_if_nodecode("tensorflow")
    DECODERS = {
        "tensorflow.sparse_tensor": _json_to_sparse_tensor,
        "tensorflow.tensor": _json_to_tensor,
        "tensorflow.variable": _json_to_variable,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a tensorflow object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `tf.RaggedTensor`: Not supported.

    - `tf.SparseTensor`:

            {
                "__type__": "tensorflow.sparse_tensor",
                "__version__": 2,
                "indices": {...},
                "values": {...},
                "shape": {...},
            }

      where the first two `{...}` placeholders result in the serialization of
      `tf.Tensor` (see below).

    - other `tf.Tensor` subtypes:

            {
                "__type__": "tensorflow.tensor",
                "__version__": 3,
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
                "__type__": "tensorflow.tensor",
                "__version__": 3,
                "id": <UUID4 str>,
            }

      By default, `TB_MAX_NBYTES` is `8000` bytes, which should be enough
      to store an array of 1000 `float64`s, and `TB_ARTIFACT_PATH` is `./`.
      `TB_ARTIFACT_PATH` must point to an existing directory.

    - `tf.Variable`:

            {
                "__type__": "tensorflow.tensor",
                "__version__": 3,
                "name": <str>,
                "value": {...},
                "trainable": <bool>,
            }

      where `{...}` is the document produced by serializing the value tensor of
      the variable.

    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (tf.RaggedTensor, _ragged_tensor_to_json),
        (tf.SparseTensor, _sparse_tensor_to_json),
        (tf.Tensor, _tensor_to_json),
        (tf.Variable, _variable_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
