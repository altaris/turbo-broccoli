"""Dataclass serialization"""

from typing import Any

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_dataclass_v3(dct: dict, ctx: Context) -> Any:
    class_name = dct["__type__"].split(".")[-1]
    return ctx.dataclass_types[class_name](**dct["data"])


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        3: _json_to_dataclass_v3,
    }
    try:
        return DECODERS[dct["__version__"]](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a dataclass into JSON by cases. The return dict has the
    following structure

    ```py
    {
        "__type__": "dataclass.<CLASS NAME>",
        "__version__": 3,
        "class": <str>,
        "data": {...},
    }
    ```

    where the `{...}` is `obj.__dict__`.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {
            "__type__": "dataclass." + obj.__class__.__name__,
            "__version__": 3,
            "data": obj.__dict__,
        }
    raise TypeNotSupported()
