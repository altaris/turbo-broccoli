"""Pytorch (de)serialization utilities."""

from typing import Any, Callable, Tuple

import safetensors.torch as st
from torch import Tensor
from torch.nn import Module
from torch.utils.data import ConcatDataset, StackDataset, Subset, TensorDataset

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _concatdataset_to_json(obj: ConcatDataset, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.concatdataset",
        "__version__": 1,
        "datasets": obj.datasets,
    }


def _json_to_concatdataset(dct: dict, ctx: Context) -> ConcatDataset:
    DECODERS = {1: _json_to_concatdataset_v1}
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_module(dct: dict, ctx: Context) -> Module:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        3: _json_to_module_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_module_v3(dct: dict, ctx: Context) -> Module:
    parts = dct["__type__"].split(".")
    type_name = ".".join(parts[2:])  # remove "pytorch.module." prefix
    module: Module = ctx.pytorch_module_types[type_name]()
    state = st.load(dct["state"])
    module.load_state_dict(state)
    return module


def _json_to_concatdataset_v1(dct: dict, ctx: Context) -> ConcatDataset:
    return ConcatDataset(dct["datasets"])


def _json_to_stackdataset(dct: dict, ctx: Context) -> StackDataset:
    DECODERS = {1: _json_to_stackdataset_v1}
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_stackdataset_v1(dct: dict, ctx: Context) -> StackDataset:
    d = dct["datasets"]
    if isinstance(d, dict):
        return StackDataset(**d)
    return StackDataset(*d)


def _json_to_subset(dct: dict, ctx: Context) -> Subset:
    DECODERS = {1: _json_to_subset_v1}
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_subset_v1(dct: dict, ctx: Context) -> Subset:
    return Subset(dct["dataset"], dct["indices"])


def _json_to_tensor(dct: dict, ctx: Context) -> Tensor:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        3: _json_to_tensor_v3,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tensor_v3(dct: dict, ctx: Context) -> Tensor:
    data = dct["data"]
    return Tensor() if data is None else st.load(data)["data"]


def _json_to_tensordataset(dct: dict, ctx: Context) -> TensorDataset:
    DECODERS = {1: _json_to_tensordataset_v1}
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tensordataset_v1(dct: dict, ctx: Context) -> TensorDataset:
    return TensorDataset(*dct["tensors"])


def _module_to_json(module: Module, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.module." + module.__class__.__name__,
        "__version__": 3,
        "state": st.save(module.state_dict()),
    }


def _stackdataset_to_json(obj: StackDataset, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.stackdataset",
        "__version__": 1,
        "datasets": obj.datasets,
    }


def _subset_to_json(obj: Subset, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.subset",
        "__version__": 1,
        "dataset": obj.dataset,
        "indices": obj.indices,
    }


def _tensor_to_json(tens: Tensor, ctx: Context) -> dict:
    x = tens.detach().cpu().contiguous()
    return {
        "__type__": "pytorch.tensor",
        "__version__": 3,
        "data": st.save({"data": x}) if x.numel() > 0 else None,
    }


def _tensordataset_to_json(obj: TensorDataset, ctx: Context) -> dict:
    return {
        "__type__": "pytorch.tensordataset",
        "__version__": 1,
        "tensors": obj.tensors,
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "pytorch.concatdataset": _json_to_concatdataset,
        "pytorch.stackdataset": _json_to_stackdataset,
        "pytorch.subset": _json_to_subset,
        "pytorch.tensor": _json_to_tensor,
        "pytorch.tensordataset": _json_to_tensordataset,
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

        ```py
        {
            "__type__": "pytorch.tensor",
            "__version__": 3,
            "data": {
                "__type__": "bytes",
                ...
            },
        }
        ```

      see `turbo_broccoli.custom.bytes.to_json`.

    - Module:

        ```py
        {
            "__type__": "pytorch.module.<class name>",
            "__version__": 3,
            "state": {
                "__type__": "bytes",
                ...
            },
        }
        ```

      see `turbo_broccoli.custom.bytes.to_json`.

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (Module, _module_to_json),
        (Tensor, _tensor_to_json),
        (ConcatDataset, _concatdataset_to_json),
        (StackDataset, _stackdataset_to_json),
        (Subset, _subset_to_json),
        (TensorDataset, _tensordataset_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
