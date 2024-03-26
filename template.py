"""Write me"""

# TODO: Write module documentation
# TODO: Replace all the XXX placeholders
# TODO: Register in custom/__init__.py (get_encoders, get_decoders)
# TODO: Document in README (Supported types)
# TODO: Document in README (nodecode types)

from typing import Any, Callable, Tuple

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _XXX_to_json(obj: Any, ctx: Context) -> dict:
    # TODO: Rename
    # TODO: Type argument
    # TODO: Write body =)
    pass


def _json_to_XXX(dct: dict, ctx: Context) -> Any:
    # TODO: Rename
    # TODO: Type return
    # TODO: Check dispatch
    DECODERS = {
        1: _json_to_XXX_v1,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_XXX_v1(dct: dict, ctx: Context) -> Any:
    # TODO: Rename
    # TODO: Type return
    # TODO: Write body =)
    pass


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    # TODO: Check dispatch
    DECODERS = {
        "XXX": _json_to_XXX,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a XXX into JSON by cases. See the README for the precise list of
    supported types.

    The return dict has the following structure

    ```py
    {
        "__type__": "XXX",
        "__version__": 1,
        ...
    }
    ```

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.
    """
    # TODO: Write doc
    # TODO: Check dispatch
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (XXX, _XXX_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
