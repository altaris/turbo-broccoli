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


def _json_to_tuple(dct: dict, ctx: Context) -> tuple:
    DECODERS = {
        1: _json_to_tuple_v1,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_tuple_v1(dct: dict, ctx: Context) -> Any:
    return tuple(dct["data"])


def _set_to_json(obj: set, ctx: Context) -> dict:
    return {"__type__": "collections.set", "__version__": 2, "data": list(obj)}


def _tuple_to_json(obj: tuple, ctx: Context) -> dict:
    """
    Converts a tuple or namedtuple into a JSON document.

    A tuple is a namedtuple if it has the following attributes: `_asdict`,
    `_field_defaults`, `_fields`, `_make`, `_replace`. See
    https://docs.python.org/3/library/collections.html#collections.namedtuple .
    """
    attributes = ["_asdict", "_field_defaults", "_fields", "_make", "_replace"]
    if not all(map(lambda a: hasattr(obj, a), attributes)):
        return {
            "__type__": "collections.tuple",
            "__version__": 1,
            "data": list(obj),
        }
    return {
        "__type__": "collections.namedtuple",
        "__version__": 2,
        "class": obj.__class__.__name__,
        "data": obj._asdict(),  # type: ignore
    }


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "collections.deque": _json_to_deque,
        "collections.namedtuple": _json_to_namedtuple,
        "collections.set": _json_to_set,
        "collections.tuple": _json_to_tuple,
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

        ```py
        {
            "__type__": "collections.deque",
            "__version__": 2,
            "data": [...],
            "maxlen": <int or None>,
        }
        ```

    - `collections.namedtuple`

        ```py
        {
            "__type__": "collections.namedtuple",
            "__version__": 2,
            "class": <str>,
            "data": {...},
        }
        ```

    - `set`

        ```py
        {
            "__type__": "collections.set",
            "__version__": 2,
            "data": [...],
        }
        ```

    - `tuple`

        ```py
        {
            "__type__": "collections.tuple",
            "__version__": 1,
            "data": [...],
        }
        ```

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (deque, _deque_to_json),
        (tuple, _tuple_to_json),
        (set, _set_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
