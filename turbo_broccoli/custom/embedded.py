"""
Embedded JSON documents.

Serializing a `EmbeddedDict` or a `EmbeddedList` will (unconditionally) result
in its own JSON artefact being created and referenced by the main JSON
document.
"""

# pylint: disable=cyclic-import
# pylint: disable=import-outside-toplevel


from typing import Any, Callable, Tuple

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


class EmbeddedDict(dict):
    """See module documentation"""


class EmbeddedList(list):
    """See module documentation"""


def _embedded_dict_to_json(obj: EmbeddedDict, ctx: Context) -> dict:
    from turbo_broccoli.turbo_broccoli import to_json as _to_json

    path, name = ctx.new_artifact_path(extension="json")
    with path.open("w", encoding="utf-8") as fp:
        fp.write(_to_json(dict(obj), ctx))
    return {"__type__": "embedded.dict", "__version__": 1, "id": name}


# TODO: deduplicate with _embedded_dict_to_json
def _embedded_list_to_json(obj: EmbeddedList, ctx: Context) -> dict:
    from turbo_broccoli.turbo_broccoli import to_json as _to_json

    path, name = ctx.new_artifact_path(extension="json")
    with path.open("w", encoding="utf-8") as fp:
        fp.write(_to_json(list(obj), ctx))
    return {"__type__": "embedded.list", "__version__": 1, "id": name}


def _json_to_embedded_dict(dct: dict, ctx: Context) -> EmbeddedDict:
    DECODERS = {
        1: _json_to_embedded_dict_v1,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_embedded_dict_v1(dct: dict, ctx: Context) -> EmbeddedDict:
    from turbo_broccoli.turbo_broccoli import from_json as _from_json

    path = ctx.id_to_artifact_path(dct["id"], extension="json")
    with path.open("r", encoding="utf-8") as fp:
        return EmbeddedDict(_from_json(fp.read(), ctx))


def _json_to_embedded_list(dct: dict, ctx: Context) -> EmbeddedList:
    DECODERS = {
        1: _json_to_embedded_list_v1,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


# TODO: deduplicate with _json_to_embedded_dict_v1
def _json_to_embedded_list_v1(dct: dict, ctx: Context) -> EmbeddedList:
    from turbo_broccoli.turbo_broccoli import from_json as _from_json

    path = ctx.id_to_artifact_path(dct["id"], extension="json")
    with path.open("r", encoding="utf-8") as fp:
        return EmbeddedList(_from_json(fp.read(), ctx))


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "embedded.dict": _json_to_embedded_dict,
        "embedded.list": _json_to_embedded_list,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a `EmbeddedDict` or an `EmbeddedList` into JSON. The return dict
    has the following structure

    ```py
    {
        "__type__": "embedded.dict",
        "__version__": 1,
        "id": <uuid4>,
    }
    ```

    or

    ```py
    {
        "__type__": "embedded.list",
        "__version__": 1,
        "id": <uuid4>,
    }
    ```

    where the UUID points to the artefact containing the actual data.
    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (EmbeddedDict, _embedded_dict_to_json),
        (EmbeddedList, _embedded_list_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
