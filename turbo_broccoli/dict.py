"""Serialization of dicts with non string keys"""
__docformat__ = "google"

from typing import Any

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_dict_v2(dct: dict) -> dict:
    """
    Converts a JSON document representing a dict with non-string keys to a dict
    following the v2 specification. In particular, note that `dct` must not
    contain the key `__dict__`.
    """
    return {item["key"]: item["val"] for item in dct["items"]}


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a non-string key dict. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__dict__`.
    """
    raise_if_nodecode("dict")
    DECODERS = {
        # 1: _json_to_dict_v1,  # Use turbo_broccoli v3
        2: _json_to_dict_v2,
    }
    try:
        return DECODERS[dct["__version__"]](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a dict with non-string keys into JSON. The return dict has the
    following structure

        {
            "__type__": "dict",
            "__version__": 2,
            "items": [
                {"key": ..., "val": ...},
                ...
            ]
        }

    """
    if not isinstance(obj, dict):
        raise TypeNotSupported()
    if all(map(lambda x: isinstance(x, str), obj.keys())):
        raise TypeNotSupported("All keys are strings")
    return {
        "__type__": "dict",
        "__version__": 2,
        "items": [{"key": key, "val": val} for key, val in obj.items()],
    }
