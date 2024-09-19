"""[`pathlib`](https://docs.python.org/3/library/pathlib.html) classes"""

from pathlib import Path
from typing import Any, Callable, Tuple

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _path_to_json(obj: Path, ctx: Context) -> dict:
    return {"__type__": "pathlib.path", "__version__": 1, "path": str(obj)}


def _json_to_path(dct: dict, ctx: Context) -> Path:
    decoders = {
        1: _json_to_path_v1,
    }
    return decoders[dct["__version__"]](dct, ctx)


def _json_to_path_v1(dct: dict, ctx: Context) -> Path:
    return Path(dct["path"])


def from_json(dct: dict, ctx: Context) -> Any:
    decoders = {
        "pathlib.path": _json_to_path,
    }
    try:
        type_name = dct["__type__"]
        return decoders[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a [`pathlib`](https://docs.python.org/3/library/pathlib.html)
    into JSON by cases. See the README for the precise list of supported types.

    The return dict has the following structure:

    - [`pathlib.Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path)

        ```py
        {
            "__type__": "pathlib.path",
            "__version__": 1,
            "path": <str>
        }
        ```

    """
    encoders: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (Path, _path_to_json),
    ]
    for t, f in encoders:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
