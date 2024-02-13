"""Python standard collections and container types (de)serialization"""

from collections import deque, namedtuple
from typing import Any, Callable, Tuple

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _deque_to_json(deq: deque, ctx: Context) -> dict:
    return {
        "__type__": "collections.deque",
        "__version__": 2,
        "data": list(deq),
        "maxlen": deq.maxlen,
    }


def _json_to_deque(dct: dict, ctx: Context) -> deque | None:
    DECODERS = {
        2: _json_to_deque_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_deque_v2(dct: dict, ctx: Context) -> Any:
    return deque(dct["data"], dct["maxlen"])


def _json_to_namedtuple(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        2: _json_to_namedtuple_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_namedtuple_v2(dct: dict, ctx: Context) -> Any:
    return namedtuple(dct["class"], dct["data"].keys())(**dct["data"])


def _json_to_set(dct: dict, ctx: Context) -> set:
    DECODERS = {
        2: _json_to_set_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_set_v2(dct: dict, ctx: Context) -> Any:
    return set(dct["data"])


def _namedtuple_to_json(tup: tuple, ctx: Context) -> dict:
    """
    Converts a namedtuple into a JSON document. This method makes sure that the
    `tup` argument is truly a namedtuple by checking that it has the following
    attributes: `_asdict`, `_field_defaults`, `_fields`, `_make`, `_replace`.
    See
    https://docs.python.org/3/library/collections.html#collections.namedtuple .
    """
    attributes = ["_asdict", "_field_defaults", "_fields", "_make", "_replace"]
    if not all(map(lambda a: hasattr(tup, a), attributes)):
        raise TypeNotSupported(
            "This object does not have all the attributes expected from a "
            "namedtuple. The expected attributes are `_asdict`, "
            "`_field_defaults`, `_fields`, `_make`, and `_replace`."
        )
    return {
        "__type__": "collections.namedtuple",
        "__version__": 2,
        "class": tup.__class__.__name__,
        "data": tup._asdict(),  # type: ignore
    }


def _set_to_json(obj: set, ctx: Context) -> dict:
    return {"__type__": "collections.set", "__version__": 2, "data": list(obj)}


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "collections.deque": _json_to_deque,
        "collections.namedtuple": _json_to_namedtuple,
        "collections.set": _json_to_set,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a Python collection into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `collections.deque`:

        ```json
        {
            "__type__": "collections.deque",
            "__version__": 2,
            "data": [...],
            "maxlen": <int or None>,
        }
        ```

    - `collections.namedtuple`

        ```json
        {
            "__type__": "collections.namedtuple",
            "__version__": 2,
            "class": <str>,
            "data": {...},
        }
        ```

    - `set`

        ```json
        {
            "__type__": "collections.set",
            "__version__": 2,
            "data": [...],
        }
        ```


    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (deque, _deque_to_json),
        (tuple, _namedtuple_to_json),
        (set, _set_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
