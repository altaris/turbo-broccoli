"""bytes (de)serialization utilities."""
__docformat__ = "google"

from base64 import b64decode, b64encode
from typing import Any, Optional

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _bytes_from_json_v1(dct: dict) -> bytes:
    """
    Deserializes a dict into a bytes object following the v1 specification.
    """
    return b64decode(dct["data"])


def from_json(dct: dict) -> Optional[bytes]:
    """
    Deserializes a dict into a bytes object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__bytes__`.
    """
    raise_if_nodecode("bytes")
    DECODERS = {
        1: _bytes_from_json_v1,
    }
    try:
        return DECODERS[dct["__bytes__"]["__version__"]](dct["__bytes__"])
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a Python `bytes` object into JSON using a base 64 + ASCII scheme.

    The return dict has the following structure

        {
            "__bytes__": {
                "__version__": 1,
                "data": <str>,
            },
        }

    """
    if isinstance(obj, bytes):
        return {
            "__bytes__": {
                "__version__": 1,
                "data": b64encode(obj).decode("ascii"),
            },
        }
    raise TypeNotSupported()
