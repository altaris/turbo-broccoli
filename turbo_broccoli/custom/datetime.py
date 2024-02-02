"""
Python datetime objects (de)serialization

See also:
    https://docs.python.org/3/library/datetime.html
"""

from datetime import datetime, time, timedelta
from typing import Any, Callable, Tuple

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _datetime_to_json(obj: datetime) -> dict:
    return {
        "__type__": "datetime.datetime",
        "__version__": 1,
        "datetime": obj.isoformat(),
    }


def _time_to_json(obj: time) -> dict:
    return {
        "__type__": "datetime.time",
        "__version__": 1,
        "time": obj.isoformat(),
    }


def _timedelta_to_json(obj: timedelta) -> dict:
    return {
        "__type__": "datetime.timedelta",
        "__version__": 1,
        "days": obj.days,
        "microseconds": obj.microseconds,
        "seconds": obj.seconds,
    }


def _json_to_datetime(dct: dict) -> datetime:
    DECODERS = {
        1: _json_to_datetime_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_datetime_v1(dct: dict) -> datetime:
    return datetime.fromisoformat(dct["datetime"])


def _json_to_time(dct: dict) -> time:
    DECODERS = {
        1: _json_to_time_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_time_v1(dct: dict) -> time:
    return time.fromisoformat(dct["time"])


def _json_to_timedelta(dct: dict) -> timedelta:
    DECODERS = {
        1: _json_to_timedelta_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_timedelta_v1(dct: dict) -> timedelta:
    return timedelta(
        days=dct["days"],
        microseconds=dct["microseconds"],
        seconds=dct["seconds"],
    )


# pylint: disable=missing-function-docstring
def from_json(dct: dict) -> Any:
    raise_if_nodecode("datetime")
    DECODERS = {
        "datetime.datetime": _json_to_datetime,
        "datetime.time": _json_to_time,
        "datetime.timedelta": _json_to_timedelta,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a XXX into JSON by cases. See the README for the precise list of
    supported types. The return dict has the following structure:

    - `datetime.datetime`:

        {
            "__type__": "datetime.datetime",
            "__version__": 1,
            "datetime": <ISO format>,
        }

    - `datetime.time`:

        {
            "__type__": "datetime.time",
            "__version__": 1,
            "time": <ISO format>,
        }

    - `datetime.timedelta`:

        {
            "__type__": "datetime.timedelta",
            "__version__": 1,
            "days": <int>,
            "microseconds": <int>,
            "seconds": <int>,
        }
    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (datetime, _datetime_to_json),
        (time, _time_to_json),
        (timedelta, _timedelta_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
