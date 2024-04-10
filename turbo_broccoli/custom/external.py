"""
External data. Embeds data from another file (decodable using
`turbo_broccoli.native.load`) into a JSON document. The data is wrapped in a
`ExternalData` objects. Note that the path of a `ExternalData` is relative to
the path of the JSON file containing it. Therefore, it is not possible to
serialize/deserialize a `ExternalData` object without a context that points to
an actual JSON file.

Warning:
    The data is read-only. Modifying `ExternalData.data` will not affect the
    data file.
"""

# pylint: disable=cyclic-import
# pylint: disable=import-outside-toplevel  # to avoid actual circular imports


from pathlib import Path
from typing import Any

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


class ExternalData:
    """Encapsulate the data of a file"""

    path: Path
    data: Any

    def __init__(self, path: Path | str, ctx: Context) -> None:
        """
        Args:
            path (Path | str): Path of the data file. Either absolute or
                relative to `ctx.file_path`
            ctx (Context):
        """
        from turbo_broccoli.native import load as native_load

        if ctx.file_path is None:
            raise ValueError("Context must have a file path")
        path = Path(path) if isinstance(path, str) else path
        if path.is_absolute():
            self.path = Path(path).relative_to(ctx.file_path.parent.absolute())
        else:
            self.path = path
        self.data = native_load(ctx.file_path.parent / self.path)


def _json_to_externaldata(dct: dict, ctx: Context) -> ExternalData:
    decoders = {
        1: _json_to_externaldata_v1,
    }
    return decoders[dct["__version__"]](dct, ctx)


def _json_to_externaldata_v1(dct: dict, ctx: Context) -> ExternalData:
    return ExternalData(dct["path"], ctx)


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> ExternalData:
    decoders = {
        "external": _json_to_externaldata,
    }
    try:
        type_name = dct["__type__"]
        return decoders[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes an `ExternalData` object into JSON. The return dict has the
    following structure

    ```py
    {
        "__type__": "external",
        "__version__": 1,
        "path": <str>
    }
    ```

    where the `path` is relative to the path of the input/output JSON file.
    """
    if not isinstance(obj, ExternalData):
        raise TypeNotSupported()
    return {
        "__type__": "external",
        "__version__": 1,
        "path": str(obj.path),
    }
