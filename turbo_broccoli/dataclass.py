"""Dataclass serialization"""
__docformat__ = "google"

from typing import Any

from turbo_broccoli.environment import (
    get_registered_dataclass_type,
)
from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _json_to_dataclass_v3(dct: dict) -> Any:
    """
    Converts a JSON document following the v2 specification to a dataclass
    object.
    """
    class_name = dct["__type__"].split(".")[-1]
    return get_registered_dataclass_type(class_name)(**dct["data"])


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a dataclass object. See `to_json` for the
    specification `dct` is expected to follow, and
    `turbo_broccoli.environment.register_dataclass_type`.
    """
    raise_if_nodecode("dataclass")
    DECODERS = {
        # 2: _json_to_dataclass_v2,  # Use turbo_broccoli v3
        3: _json_to_dataclass_v3,
    }
    try:
        raise_if_nodecode(dct["__type__"])
        return DECODERS[dct["__version__"]](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a dataclass into JSON by cases. The return dict has the
    following structure

        {
            "__type__": "dataclass.<CLASS NAME>",
            "__version__": 3,
            "class": <str>,
            "data": {...},
        }

    where the `{...}` is `obj.__dict__`.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {
            "__type__": "dataclass." + obj.__class__.__name__,
            "__version__": 3,
            "data": obj.__dict__,
        }
    raise TypeNotSupported()
