"""bytes (de)serialization utilities."""

from base64 import b64decode, b64encode
from math import ceil
from typing import Any

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _bytes_from_json_v3(dct: dict, ctx: Context) -> bytes:
    if "data" in dct:
        return b64decode(dct["data"])
    path = ctx.id_to_artifact_path(dct["id"])
    with path.open(mode="rb") as fp:
        return fp.read()


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> bytes | None:
    DECODERS = {
        3: _bytes_from_json_v3,
    }
    try:
        return DECODERS[dct["__version__"]](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a Python `bytes` object into JSON using a base64 + ASCII
    scheme. The return dict has the following structure

    ```py
    {
        "__type__": "bytes",
        "__version__": 3,
        "data": <ASCII str>,
    }
    ```

    or

    ```py
    {
        "__type__": "bytes",
        "__version__": 3,
        "id": <uuid4>,
    }
    ```

    if the base64 encoding of the object is too large.

    """
    if not isinstance(obj, bytes):
        raise TypeNotSupported()
    # https://stackoverflow.com/a/32140193
    b64_size = (ceil((len(obj) * 4) / 3) + 3) & ~3
    if b64_size <= ctx.min_artifact_size:
        return {
            "__type__": "bytes",
            "__version__": 3,
            "data": b64encode(obj).decode("ascii"),
        }
    path, name = ctx.new_artifact_path()
    with path.open(mode="wb") as fp:
        fp.write(obj)
    return {"__type__": "bytes", "__version__": 3, "id": name}
