"""Write me"""
__docformat__ = "google"

# TODO: Write module documentation
# TODO: Replace all the XXX placeholders
# TODO: Register in main module
# TODO: Document in README

from typing import Any, Callable, List, Tuple


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
    DECODERS = {
        1: _json_to_XXX_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_XXX_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a XXX following the v1 specification.
    """
    # TODO: Rename
    # TODO: Type argument


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a XXX. See `to_json` for the specification `dct`
    is expected to follow. In particular, note that `dct` must contain the key
    `__XXX__`.
    """
    DECODERS = {
        "XXX": _json_to_XXX,
    }
    try:
        return DECODERS[dct["__XXX__"]["__type__"]](dct["__XXX__"])
    except KeyError as exc:
        raise TypeError("Not a valid XXX document") from exc


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
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (XXX, _XXX_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__XXX__": f(obj)}
    raise TypeError("Not a supported XXX type")
