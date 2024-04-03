"""
UUIDs

See also:
    https://docs.python.org/3/library/uuid.html
"""

from typing import Any
from uuid import UUID

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _json_to_uuid_v1(dct: dict, ctx: Context) -> Any:
    return UUID(hex=dct["hex"])


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    try:
        DECODERS = {
            1: _json_to_uuid_v1,
        }
        return DECODERS[dct["__version__"]](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a
    [`uuid.UUID`](https://docs.python.org/3/library/uuid.html#uuid.UUID) object
    into JSON. The return dict has the following structure

    ```py
    {
        "__type__": "uuid",
        "__version__": 1,
        "hex": <str>
    }
    ```

    where `<hex>` is the hexadecimal representation of the UUID. The reason for
    using this representation instead of others (e.g. int or bytes) is that in
    string form, this is the shortest.
    """
    if not isinstance(obj, UUID):
        raise TypeNotSupported()
    return {"__type__": "uuid", "__version__": 1, "hex": obj.hex}
