"""Tensorflow (de)serialization utilities."""

from typing import Any, Callable, Tuple

import tensorflow as tf
from safetensors import tensorflow as st

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_sparse_tensor(dct: dict, ctx: Context) -> tf.Tensor:
    DECODERS = {
        2: _json_to_sparse_tensor_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_sparse_tensor_v2(dct: dict, ctx: Context) -> tf.Tensor:
    return tf.SparseTensor(
        dense_shape=dct["shape"],
        indices=dct["indices"],
        values=dct["values"],
    )


def _json_to_tensor(dct: dict, ctx: Context) -> tf.Tensor:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        4: _json_to_tensor_v4,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tensor_v4(dct: dict, ctx: Context) -> tf.Tensor:
    return st.load(dct["data"])["data"]


def _json_to_variable(dct: dict, ctx: Context) -> tf.Variable:
    DECODERS = {
        3: _json_to_variable_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_variable_v3(dct: dict, ctx: Context) -> tf.Variable:
    return tf.Variable(
        initial_value=dct["value"],
        name=dct["name"],
        trainable=dct["trainable"],
    )


def _ragged_tensor_to_json(obj: tf.Tensor, ctx: Context) -> dict:
    raise NotImplementedError(
        "Serialization of ragged tensors is not supported"
    )


def _sparse_tensor_to_json(obj: tf.SparseTensor, ctx: Context) -> dict:
    return {
        "__type__": "tensorflow.sparse_tensor",
        "__version__": 2,
        "indices": obj.indices,
        "shape": list(obj.dense_shape),
        "values": obj.values,
    }


def _tensor_to_json(obj: tf.Tensor, ctx: Context) -> dict:
    return {
        "__type__": "tensorflow.tensor",
        "__version__": 4,
        "data": st.save({"data": obj}),
    }


def _variable_to_json(var: tf.Variable, ctx: Context) -> dict:
    return {
        "__type__": "tensorflow.variable",
        "__version__": 3,
        "name": var.name,
        "value": var.value(),
        "trainable": var.trainable,
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "tensorflow.sparse_tensor": _json_to_sparse_tensor,
        "tensorflow.tensor": _json_to_tensor,
        "tensorflow.variable": _json_to_variable,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a tensorflow object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `tf.RaggedTensor`: Not supported.

    - `tf.SparseTensor`:

        ```py
        {
            "__type__": "tensorflow.sparse_tensor",
            "__version__": 2,
            "indices": {...},
            "values": {...},
            "shape": {...},
        }
        ```

      where the first two `{...}` placeholders result in the serialization of
      `tf.Tensor` (see below).

    - other `tf.Tensor` subtypes:

        ```py
        {
            "__type__": "tensorflow.tensor",
            "__version__": 4,
            "data": {
                "__type__": "bytes",
                ...
            },
        }
        ```

      see `turbo_broccoli.custom.bytes.to_json`.

    - `tf.Variable`:

        ```py
        {
            "__type__": "tensorflow.tensor",
            "__version__": 3,
            "name": <str>,
            "value": {...},
            "trainable": <bool>,
        }
        ```

      where `{...}` is the document produced by serializing the value tensor of
      the variable, see above.

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (tf.RaggedTensor, _ragged_tensor_to_json),
        (tf.SparseTensor, _sparse_tensor_to_json),
        (tf.Tensor, _tensor_to_json),
        (tf.Variable, _variable_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
