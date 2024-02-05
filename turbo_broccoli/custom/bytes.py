"""bytes (de)serialization utilities."""

from base64 import b64decode, b64encode
from typing import Any

from turbo_broccoli.context import Context
from turbo_broccoli.utils import DeserializationError, TypeNotSupported


def _bytes_from_json_v2(dct: dict, ctx: Context) -> bytes:
    return b64decode(dct["data"])


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> bytes | None:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        # 1: _bytes_from_json_v2,  # Use turbo_broccoli v3
        2: _bytes_from_json_v2,
    }
    try:
        return DECODERS[dct["__version__"]](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a Python `bytes` object into JSON using a base 64 + ASCII
    scheme. The return dict has the following structure

        {
            "__type__": "bytes",
            "__version__": 2,
            "data": <ASCII str>,
        }

    """
    if isinstance(obj, bytes):
        return {
            "__type__": "bytes",
            "__version__": 2,
            "data": b64encode(obj).decode("ascii"),
        }
    raise TypeNotSupported()
