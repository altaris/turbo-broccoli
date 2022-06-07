"""Tensorflow (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
except ModuleNotFoundError:
    HAS_TENSORFLOW = False


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


def _ragged_tensor_to_json(tens: tf.Tensor) -> dict:
    """Serializes a general tensor"""
    raise NotImplementedError(
        "Serialization of ragged tensors is not supported"
    )


def _tensor_to_json(tens: tf.Tensor) -> dict:
    """Serializes a general tensor"""
    return {
        "__type__": "tensor",
        "__version__": 1,
        "dtype": tens.dtype.name,
        "numpy": tens.numpy(),
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a tensorflow object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__tensorflow__`.
    """
    DECODERS = {
        "tensor": _json_to_tensor,
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
    * `tf.SparseTensor`: Will be treated like general tensors.
    * other `tf.Tensor` subtypes:

            {
                "__type__": "tensor",
                "__version__": 1,
                "dtype": <str>,
                "numpy": {...},
            }

      where `{...}` is the document produced by `turbo_broccoli.numpy.to_json`.

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (tf.RaggedTensor, _ragged_tensor_to_json),
        (tf.Tensor, _tensor_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__tensorflow__": f(obj)}
    raise TypeError("Not a supported tensorflow type")
