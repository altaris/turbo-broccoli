"""Write me"""
__docformat__ = "google"

# TODO: Write module documentation
# TODO: Replace all the XXX placeholders
# TODO: Register in main module
# TODO: Document in README (Supported types)
# TODO: Document in README (nodecode types)

from typing import Any, Callable, Tuple

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _XXX_to_json(obj: Any) -> dict:
    """Converts a XXX into a JSON document."""
    # TODO: Rename
    # TODO: Type argument
    # TODO: Write body =)


def _json_to_XXX(dct: dict) -> Any:
    """
    Converts a JSON document to a XXX. See `to_json` for the specification
    `dct` is expected to follow. Note that the key `__XXX__` should not be
    present.
    """
    # TODO: Rename
    # TODO: Type return
    # TODO: Check dispatch
    DECODERS = {
        1: _json_to_XXX_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_XXX_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a XXX following the v1 specification.
    """
    # TODO: Rename
    # TODO: Type return
    # TODO: Write body =)


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a XXX. See `to_json` for the specification `dct`
    is expected to follow. In particular, note that `dct` must contain the key
    `__XXX__`.
    """
    # TODO: Check dispatch
    raise_if_nodecode("XXX")
    DECODERS = {
        "XXX": _json_to_XXX,
    }
    try:
        type_name = dct["__XXX__"]["__type__"]
        raise_if_nodecode("XXX." + type_name)
        return DECODERS[type_name](dct["__XXX__"])
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a XXX into JSON by cases. See the README for the precise list of
    supported types.

    The return dict has the following structure

        {
            "__XXX__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.
    """
    # TODO: Write doc
    # TODO: Check dispatch
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (XXX, _XXX_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__XXX__": f(obj)}
    raise TypeNotSupported()
