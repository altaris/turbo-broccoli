"""Bokeh objects (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, Tuple

from bokeh.core.serialization import (
    Buffer,
    Deserializer,
    Serialized,
    Serializer,
)
from bokeh.models import Model
from bokeh.plotting import figure as Figure

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _buffer_to_json(obj: Buffer) -> dict:
    """Serializes a bokeh object.into a JSON document."""
    return {
        "__type__": "bokeh.buffer",
        "__version__": 2,
        "id": obj.id,
        "data": obj.to_bytes(),
    }


def _generic_to_json(obj: Figure) -> dict:
    """Serializes a bokeh object.into a JSON document."""
    s = Serializer().serialize(obj)
    return {
        "__type__": "bokeh.generic",
        "__version__": 2,
        "content": s.content,
        "buffers": s.buffers,
    }


def _json_to_buffer(dct: dict) -> Buffer:
    """
    Converts a JSON document to a bokeh buffer. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__bokeh__`
    should not be present.
    """
    DECODERS = {
        # 1: _json_to_buffer_v1,  # Use turbo_broccoli v3
        2: _json_to_buffer_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_buffer_v2(dct: dict) -> Buffer:
    """
    Converts a JSON document to a bokeh buffer following the v1 specification.
    """
    return Buffer(id=dct["id"], data=dct["data"])


def _json_to_generic(dct: dict) -> Any:
    """
    Converts a JSON document Serializes a bokeh object. See `to_json` for
    the specification `dct` is expected to follow. Note that the key
    `__bokeh__` should not be present.
    """
    DECODERS = {
        # 1: _json_to_buffer_v1,  # Use turbo_broccoli v3
        2: _json_to_generic_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_generic_v2(dct: dict) -> Any:
    """
    Converts a JSON document Serializes a bokeh object.following the v1
    specification.
    """
    c, b = dct["content"], dct["buffers"]
    return Deserializer().deserialize(Serialized(content=c, buffers=b))


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict Serializes a bokeh object. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__bokeh__`.
    """
    raise_if_nodecode("bokeh")
    DECODERS = {
        "bokeh.buffer": _json_to_buffer,
        "bokeh.generic": _json_to_generic,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a bokeh object.

    The return dict has the following structure

    - `bokeh.plotting._figure.figure` or `bokeh.models.Model`:

            {
                "__type__": "bokeh.generic",
                "__version__": 2,
                "content": {...},
                "buffers": [...],
            }

    - `bokeh.core.serialization.Buffer`: (for internal use)

            {
                "__type__": "bokeh.buffer",
                "__version__": 2,
                "id": <str>,
                "data": <bytes>,
            }

    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (Buffer, _buffer_to_json),
        (Figure, _generic_to_json),
        (Model, _generic_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
