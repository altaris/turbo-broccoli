"""Python standard collections and container types (de)serialization"""

from collections import deque, namedtuple
from typing import Any, Callable, Tuple

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _deque_to_json(deq: deque) -> dict:
    return {
        "__type__": "collections.deque",
        "__version__": 2,
        "data": list(deq),
        "maxlen": deq.maxlen,
    }


def _json_to_deque(dct: dict) -> deque | None:
    DECODERS = {
        # 1: _json_to_deque_v1,  # Use turbo_broccoli v3
        2: _json_to_deque_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_deque_v2(dct: dict) -> Any:
    return deque(dct["data"], dct["maxlen"])


def _json_to_namedtuple(dct: dict) -> Any:
    DECODERS = {
        # 1: _json_to_namedtuple_v1,  # Use turbo_broccoli v3
        2: _json_to_namedtuple_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_namedtuple_v2(dct: dict) -> Any:
    return namedtuple(dct["class"], dct["data"].keys())(**dct["data"])


def _json_to_set(dct: dict) -> set:
    DECODERS = {
        # 1: _json_to_set_v1,  # Use turbo_broccoli v3
        2: _json_to_set_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_set_v2(dct: dict) -> Any:
    return set(dct["data"])


def _namedtuple_to_json(tup: tuple) -> dict:
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


def _set_to_json(obj: set) -> dict:
    return {"__type__": "collections.set", "__version__": 2, "data": list(obj)}


# pylint: disable=missing-function-docstring
def from_json(dct: dict) -> Any:
    DECODERS = {
        "collections.deque": _json_to_deque,
        "collections.namedtuple": _json_to_namedtuple,
        "collections.set": _json_to_set,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a Python collection into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - `collections.deque`:

            {
                "__type__": "collections.deque",
                "__version__": 2,
                "data": [...],
                "maxlen": <int or None>,
            }

    - `collections.namedtuple`

            {
                "__type__": "collections.namedtuple",
                "__version__": 2,
                "class": <str>,
                "data": {...},
            }

    - `set`

            {
                "__type__": "collections.set",
                "__version__": 2,
                "data": [...],
            }


    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (deque, _deque_to_json),
        (tuple, _namedtuple_to_json),
        (set, _set_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
