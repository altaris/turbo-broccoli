"""Pytorch (de)serialization utilities."""

from typing import Any, Callable, Tuple

import safetensors.torch as st
import torch

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_module(dct: dict, ctx: Context) -> torch.nn.Module:
    DECODERS = {
        3: _json_to_module_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_module_v3(dct: dict, ctx: Context) -> torch.nn.Module:
    parts = dct["__type__"].split(".")
    type_name = ".".join(parts[2:])  # remove "pytorch.module." prefix
    module: torch.nn.Module = ctx.pytorch_module_types[type_name]()
    state = st.load(dct["state"])
    module.load_state_dict(state)
    return module


def _json_to_tensor(dct: dict, ctx: Context) -> torch.Tensor:
    DECODERS = {
        3: _json_to_tensor_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tensor_v3(dct: dict, ctx: Context) -> torch.Tensor:
    data = dct["data"]
    return torch.Tensor() if data is None else st.load(data)["data"]


def _module_to_json(module: torch.nn.Module, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.module." + module.__class__.__name__,
        "__version__": 3,
        "state": st.save(module.state_dict()),
    }


def _tensor_to_json(tens: torch.Tensor, ctx: Context) -> dict:
    x = tens.detach().cpu().contiguous()
    return {
        "__type__": "pytorch.tensor",
        "__version__": 3,
        "data": st.save({"data": x}) if x.numel() > 0 else None,
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "pytorch.tensor": _json_to_tensor,
    }
    try:
        type_name = dct["__type__"]
        if type_name.startswith("pytorch.module."):
            return _json_to_module(dct, ctx)
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
                "__version__": 3,
                "data": {
                    "__type__": "bytes",
                    ...
                },
            }

      see `turbo_broccoli.custom.bytes.to_json`.

    - Module:

            {
                "__type__": "pytorch.module.<class name>",
                "__version__": 3,
                "state": {
                    "__type__": "bytes",
                    ...
                },
            }

      see `turbo_broccoli.custom.bytes.to_json`.

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (torch.nn.Module, _module_to_json),
        (torch.Tensor, _tensor_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
