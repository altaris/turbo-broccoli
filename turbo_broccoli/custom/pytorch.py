"""Pytorch (de)serialization utilities."""

from typing import Any, Callable, Tuple

import safetensors.torch as st
import torch

from turbo_broccoli.context import Context
from turbo_broccoli.utils import DeserializationError, TypeNotSupported


def _json_to_module(dct: dict, ctx: Context) -> torch.nn.Module:
    DECODERS = {
        # 1: _json_to_module_v1,  # Use turbo_broccoli v3
        2: _json_to_module_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_module_v2(dct: dict, ctx: Context) -> torch.nn.Module:
    module: torch.nn.Module = ctx.pytorch_module_types[dct["class"]]()
    module.load_state_dict(dct["state"])
    return module


def _json_to_tensor(dct: dict, ctx: Context) -> torch.Tensor:
    DECODERS = {
        # 1: _json_to_tensor_v1,  # Use turbo_broccoli v3
        2: _json_to_tensor_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tensor_v2(dct: dict, ctx: Context) -> torch.Tensor:
    if "data" in dct:
        if dct["data"] is None:  # empty tensor
            return torch.Tensor()
        return st.load(dct["data"])["data"]
    return st.load_file(ctx.artifact_path / (dct["id"] + ".tb"))["data"]


def _module_to_json(module: torch.nn.Module, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.module",
        "__version__": 2,
        "class": module.__class__.__name__,
        "state": module.state_dict(),
    }


def _tensor_to_json(tens: torch.Tensor, ctx: Context) -> dict:
    x = tens.detach().cpu().contiguous()
    if x.numel() == 0:  # empty tensor
        return {
            "__type__": "pytorch.tensor",
            "__version__": 2,
            "data": None,
        }
    if x.numpy().nbytes <= ctx.min_artifact_size:
        return {
            "__type__": "pytorch.tensor",
            "__version__": 2,
            "data": st.save({"data": x}),
        }
    path, name = ctx.new_artifact_path()
    st.save_file({"data": x}, path)
    return {"__type__": "pytorch.tensor", "__version__": 2, "id": name}


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    ctx.raise_if_nodecode("pytorch")
    DECODERS = {
        "pytorch.tensor": _json_to_tensor,
        "pytorch.module": _json_to_module,
    }
    try:
        type_name = dct["__type__"]
        ctx.raise_if_nodecode(type_name)
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
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
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (torch.nn.Module, _module_to_json),
        (torch.Tensor, _tensor_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
