"""Serialization of dicts with non string keys"""
__docformat__ = "google"

from typing import Any

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_dict_v1(dct: dict) -> dict:
    """
    Converts a JSON document representing a dict with non-string keys to a dict
    following the v1 specification. In particular, note that `dct` must not
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
        1: _json_to_dict_v1,
    }
    try:
        return DECODERS[dct["__dict__"]["__version__"]](dct["__dict__"])
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a dict with non-string keys into JSON. The return dict has the
    following structure

        {
            "__dict__": {
                "__version__": 1,
                "items": [
                    {"key": {...}, "val": {...}},
                    ...
                ]
            },
        }

    """
    if not isinstance(obj, dict):
        raise TypeNotSupported()
    if all(map(lambda x: isinstance(x, str), obj.keys())):
        raise TypeNotSupported("All keys are strings")
    return {
        "__dict__": {
            "__version__": 1,
            "items": [{"key": key, "val": val} for key, val in obj.items()],
        },
    }
