"""Python standard collections (de)serialization"""
__docformat__ = "google"

from collections import deque, namedtuple
from typing import Any, Callable, List, Tuple


def _deque_to_json(deq: deque) -> dict:
    """Converts a deque into a JSON document."""
    return {
        "__type__": "deque",
        "__version__": 1,
        "data": list(deq),
        "maxlen": deq.maxlen,
    }


def _json_to_deque(dct: dict) -> deque:
    """
    Converts a JSON document to a deque. See `to_json` for the specification
    `dct` is expected to follow. Note that the key `__collections__` should not
    be present.
    """
    DECODERS = {
        1: _json_to_deque_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_deque_v1(dct: dict) -> Any:
    """Converts a JSON document to a deque following the v1 specification."""
    return deque(dct["data"], dct["maxlen"])


def _json_to_namedtuple(dct: dict) -> Any:
    """
    Converts a JSON document to a namedtuple. See `to_json` for the
    specification `dct` is expected to follow. Note that the key
    `__collections__` should not be present.
    """
    DECODERS = {
        1: _json_to_namedtuple_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_namedtuple_v1(dct: dict) -> Any:
    """Converts a JSON document to a deque following the v1 specification."""
    return namedtuple(dct["class"], dct["data"].keys())(**dct["data"])


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
        raise TypeError("Not a namedtuple")
    return {
        "__type__": "namedtuple",
        "__version__": 1,
        "class": tup.__class__.__name__,
        "data": tup._asdict(),  # type: ignore
    }


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a Python collection. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__collections__`.
    """
    DECODERS = {
        "deque": _json_to_deque,
        "namedtuple": _json_to_namedtuple,
    }
    try:
        return DECODERS[dct["__collections__"]["__type__"]](
            dct["__collections__"]
        )
    except KeyError as exc:
        raise TypeError("Not a valid collections document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a Python collection into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__collections__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.

    * `collections.deque`:

        {
            "__collections__": {
                "__type__": "deque,
                "__version__": 1,
                "data": [...],
                "maxlen": <int or None>,
            }
        }

    * `collections.namedtuple`

        {
            "__collections__": {
                "__type__": "namedtuple,
                "__version__": 1,
                "class": <str>,
                "data": {...},
            }
        }

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (deque, _deque_to_json),
        (tuple, _namedtuple_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__collections__": f(obj)}
    raise TypeError("Not a supported collections type")
