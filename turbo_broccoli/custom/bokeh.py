"""Bokeh objects (de)serialization utilities."""

from typing import Any, Callable, Tuple

from bokeh.core.serialization import (
    Buffer,
    Deserializer,
    Serialized,
    Serializer,
)
from bokeh.models import Model
from bokeh.plotting import figure as Figure

from ..context import Context
from ..exceptions import DeserializationError, TypeNotSupported


def _buffer_to_json(obj: Buffer, ctx: Context) -> dict:
    return {
        "__type__": "bokeh.buffer",
        "__version__": 2,
        "id": obj.id,
        "data": obj.to_bytes(),
    }


def _generic_to_json(obj: Figure, ctx: Context) -> dict:
    s = Serializer().serialize(obj)
    return {
        "__type__": "bokeh.generic",
        "__version__": 2,
        "content": s.content,
        "buffers": s.buffers,
    }


def _json_to_buffer(dct: dict, ctx: Context) -> Buffer:
    decoders = {
        2: _json_to_buffer_v2,
    }
    return decoders[dct["__version__"]](dct, ctx)


def _json_to_buffer_v2(dct: dict, ctx: Context) -> Buffer:
    return Buffer(id=dct["id"], data=dct["data"])


def _json_to_generic(dct: dict, ctx: Context) -> Any:
    decoders = {
        2: _json_to_generic_v2,
    }
    return decoders[dct["__version__"]](dct, ctx)


def _json_to_generic_v2(dct: dict, ctx: Context) -> Any:
    c, b = dct["content"], dct["buffers"]
    return Deserializer().deserialize(Serialized(content=c, buffers=b))


def from_json(dct: dict, ctx: Context) -> Any:
    ctx.raise_if_nodecode("bytes")
    decoders = {
        "bokeh.buffer": _json_to_buffer,
        "bokeh.generic": _json_to_generic,
    }
    try:
        type_name = dct["__type__"]
        return decoders[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a bokeh object. The return dict has the following structure:

    - `bokeh.plotting._figure.figure` or `bokeh.models.Model`:

        ```py
        {
            "__type__": "bokeh.generic",
            "__version__": 2,
            "content": {...},
            "buffers": [...],
        }
        ```

    - `bokeh.core.serialization.Buffer`: (for internal use)

        ```py
        {
            "__type__": "bokeh.buffer",
            "__version__": 2,
            "id": <str>,
            "data": <bytes>,
        }
        ```

    """
    encoders: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (Buffer, _buffer_to_json),
        (Figure, _generic_to_json),
        (Model, _generic_to_json),
    ]
    for t, f in encoders:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
