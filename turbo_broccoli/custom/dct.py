"""Dicts with non-string keys"""

from typing import Any

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_dict_v1(dct: dict, ctx: Context) -> dict:
    return {d["key"]: d["value"] for d in dct["data"]}


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> dict:
    try:
        DECODERS = {
            1: _json_to_dict_v1,
        }
        return DECODERS[dct["__version__"]](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a dict with non-string keys. The return dict has the following
    structure

    ```py
    {
        "__type__": "dict",
        "__version__": 1,
        "data": [
            {
                "key": ...,
                "value": ...,
            },
            ...
        ]
    }
    ```

    where the keys are values are themselves converted to JSON.
    """
    if not (
        isinstance(obj, dict)
        and not all(isinstance(k, str) for k in obj.keys())
    ):
        raise TypeNotSupported()
    return {
        "__type__": "dict",
        "__version__": 1,
        "data": [{"key": k, "value": v} for k, v in obj.items()],
    }
