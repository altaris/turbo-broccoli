"""Bokeh objects (de)serialization utilities."""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple

from bokeh.plotting import figure as Figure
from bokeh.models import Model
from bokeh.core.serialization import Buffer
from bokeh.core.serialization import Deserializer, Serialized, Serializer


def _buffer_to_json(obj: Buffer) -> dict:
    """Serializes a bokeh object.into a JSON document."""
    return {
        "__type__": "buffer",
        "__version__": 1,
        "id": obj.id,
        "data": obj.to_bytes(),
    }


def _generic_to_json(obj: Figure) -> dict:
    """Serializes a bokeh object.into a JSON document."""
    s = Serializer().serialize(obj)
    return {
        "__type__": "generic",
        "__version__": 1,
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
        1: _json_to_buffer_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_buffer_v1(dct: dict) -> Buffer:
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
        1: _json_to_generic_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_generic_v1(dct: dict) -> Any:
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
    DECODERS = {
        "buffer": _json_to_buffer,
        "generic": _json_to_generic,
    }
    try:
        return DECODERS[dct["__bokeh__"]["__type__"]](dct["__bokeh__"])
    except KeyError as exc:
        raise TypeError("Not a valid figure document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a bokeh object.

    The return dict has the following structure

        {
            "__bokeh__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.

    * `bokeh.plotting._figure.figure` or `bokeh.models.Model`:

        {
            "__bokeh__": {
                "__type__": "generic",
                "__version__": 1,
                "content": {...},
                "buffers": [...],
            }
        }

    * `bokeh.core.serialization.Buffer`: (for internal use)

        {
            "__bokeh__": {
                "__type__": "buffer",
                "__version__": 1,
                "id": <str>,
                "data": <bytes>,
            }
        }

    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (Buffer, _buffer_to_json),
        (Figure, _generic_to_json),
        (Model, _generic_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__bokeh__": f(obj)}
    raise TypeError("Not a supported figure type")
