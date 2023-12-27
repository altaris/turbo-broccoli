"""Pytorch (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, Tuple
from uuid import uuid4

import safetensors.torch as st
import torch

from turbo_broccoli.environment import (
    get_artifact_path,
    get_max_nbytes,
    get_registered_pytorch_module_type,
)
from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_module(dct: dict) -> torch.nn.Module:
    """
    Converts a JSON document to a `pytorch` module. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__pytorch__`
    should not be present.
    """
    DECODERS = {
        # 1: _json_to_module_v1,  # Use turbo_broccoli v3
        2: _json_to_module_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_module_v2(dct: dict) -> torch.nn.Module:
    """
    Converts a JSON document to a `pytorch` module following the v2
    specification.
    """
    module: torch.nn.Module = get_registered_pytorch_module_type(
        dct["class"]
    )()
    module.load_state_dict(dct["state"])
    return module


def _json_to_tensor(dct: dict) -> torch.Tensor:
    """
    Converts a JSON document to a `pytorch` tensor. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__pytorch__`
    should not be present.
    """
    DECODERS = {
        # 1: _json_to_tensor_v1,  # Use turbo_broccoli v3
        2: _json_to_tensor_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_tensor_v2(dct: dict) -> torch.Tensor:
    """
    Converts a JSON document to a `pytorch` tensor following the v2
    specification.
    """
    if "data" in dct:
        if dct["data"] is None:  # empty tensor
            return torch.Tensor()
        return st.load(dct["data"])["data"]
    return st.load_file(get_artifact_path() / dct["id"])["data"]


def _module_to_json(module: torch.nn.Module) -> dict:
    """Converts a pytorch `torch.nn.Module` into a JSON document."""
    return {
        "__type__": "pytorch.module",
        "__version__": 2,
        "class": module.__class__.__name__,
        "state": module.state_dict(),
    }


def _tensor_to_json(tens: torch.Tensor) -> dict:
    """Converts a tensor into a JSON document."""
    x = tens.detach().cpu().contiguous()
    if x.numel() == 0:  # empty tensor
        return {
            "__type__": "pytorch.tensor",
            "__version__": 2,
            "data": None,
        }
    if x.numpy().nbytes <= get_max_nbytes():
        return {
            "__type__": "pytorch.tensor",
            "__version__": 2,
            "data": st.save({"data": x}),
        }
    name = str(uuid4())
    st.save_file({"data": x}, get_artifact_path() / name)
    return {
        "__type__": "pytorch.tensor",
        "__version__": 2,
        "id": name,
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a `pytorch` object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__pytorch__`.
    """
    raise_if_nodecode("pytorch")
    DECODERS = {
        "pytorch.tensor": _json_to_tensor,
        "pytorch.module": _json_to_module,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a tensor into JSON by cases. See the README for the precise list
    of supported types. The return dict has the following structure:

    - Tensor:

            {
                "__type__": "pytorch.tensor",
                "__version__": 2,
                "data": <bytes>,
            }

      or if the underlying data is too large resulting in an artifact being
      created:

            {
                "__type__": "pytorch.tensor",
                "__version__": 2,
                "id": <UUID4 str>,
            }

    - Module:

            {
                "__type__": "pytorch.module",
                "__version__": 2,
                "class": <class name>,
                "state": {...},
            }

    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (torch.nn.Module, _module_to_json),
        (torch.Tensor, _tensor_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
