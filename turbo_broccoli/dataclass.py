"""Dataclass serialization"""
__docformat__ = "google"

from typing import Any


def to_json(obj: Any) -> dict:
    """
    Serializes a dataclass into JSON by cases. The return dict has the
    following structure

        {
            "__dataclass__": {
                "__version__": 1,
                "data": {...},
            },
        }

    where the `{...}` is `obj.__dict__`.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {
            "__dataclass__": {
                "__version__": 1,
                "data": obj.__dict__,
            },
        }
    raise TypeError("Not a supported type")
