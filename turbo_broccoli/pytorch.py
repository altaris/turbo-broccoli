"""Pytorch (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple
from uuid import uuid4

from torch import Tensor
import safetensors.torch as st

from turbo_broccoli.environment import get_artifact_path, get_max_nbytes


def _json_to_tensor(dct: dict) -> Tensor:
    """
    Converts a JSON document to a `pytorch` tensor. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__tensor__`
    should not be present.
    """
    DECODERS = {
        1: _json_to_tensor_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_tensor_v1(dct: dict) -> Tensor:
    """
    Converts a JSON document to a `pytorch` tensor following the v1 specification.
    """
    if "data" in dct:
        if dct["data"] is None:  # empty tensor
            return Tensor()
        return st.load(dct["data"])["data"]
    return st.load_file(get_artifact_path() / dct["id"])["data"]


def _tensor_to_json(tens: Tensor) -> dict:
    """Converts a tensor into a JSON document."""
    x = tens.detach().cpu()
    if x.numel() == 0:  # empty tensor
        return {
            "__type__": "tensor",
            "__version__": 1,
            "data": None,
        }
    if x.numpy().nbytes <= get_max_nbytes():
        return {
            "__type__": "tensor",
            "__version__": 1,
            "data": st.save({"data": x}),
        }
    name = str(uuid4())
    st.save_file({"data": x}, get_artifact_path() / name)
    return {
        "__type__": "tensor",
        "__version__": 1,
        "id": name,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a `pytorch` object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__pytorch__`.
    """
    DECODERS = {
        "tensor": _json_to_tensor,
    }
    try:
        return DECODERS[dct["__pytorch__"]["__type__"]](dct["__pytorch__"])
    except KeyError as exc:
        raise TypeError("Not a valid pytorch document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a tensor into JSON by cases. See the README for the precise list
    of supported types.

    The return dict has the following structure

        {
            "__pytorch__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.
    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (Tensor, _tensor_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__pytorch__": f(obj)}
    raise TypeError("Not a supported tensor type")
