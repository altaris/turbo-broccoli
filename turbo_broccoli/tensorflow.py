"""Tensorflow (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple

import tensorflow as tf


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
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_tensor_v1(dct: dict) -> tf.Tensor:
    """
    Converts a JSON document following the v1 specification to a tensorflow
    tensor.
    """
    return tf.constant(dct["numpy"], dtype=dct["dtype"])


def _json_to_variable(dct: dict) -> tf.Variable:
    """Converts a JSON document to a tensorflow variable."""
    DECODERS = {
        1: _json_to_variable_v1,
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
        "shape": tens.dense_shape.numpy(),
        "values": tens.values,
    }


def _tensor_to_json(tens: tf.Tensor) -> dict:
    """Serializes a general tensor"""
    return {
        "__type__": "tensor",
        "__version__": 1,
        "dtype": tens.dtype.name,
        "numpy": tens.numpy(),
    }


def _variable_to_json(var: tf.Variable) -> dict:
    """Serializes a general tensor"""
    return {
        "__type__": "variable",
        "__version__": 1,
        "dtype": var.dtype.name,
        "name": var.name,
        "numpy": var.numpy(),
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
        return DECODERS[dct["__tensorflow__"]["__type__"]](
            dct["__tensorflow__"]
        )
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

      where `{...}` is the document produced by `turbo_broccoli.numpy.to_json`.
    * `tf.Variable`:

            {
                "__type__": "tensor",
                "__version__": 1,
                "dtype": <str>,
                "name": <str>,
                "numpy": {...},
                "trainable": <bool>,
            }

      where `{...}` is the document produced by `turbo_broccoli.numpy.to_json`.

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
