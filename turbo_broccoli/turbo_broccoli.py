"""Main module containing the JSON encoder and decoder methods."""

import json
import zlib
from pathlib import Path
from typing import Any

from . import user
from .context import Context
from .custom import get_decoders, get_encoders
from .exceptions import TypeIsNodecode, TypeNotSupported


def _from_jsonable(obj: Any, ctx: Context) -> Any:
    """
    Takes an object fresh from `json.load` or `json.loads` and loads types that
    are supported by TurboBroccoli therein.
    """
    if isinstance(obj, dict):
        obj = {k: _from_jsonable(v, ctx / k) for k, v in obj.items()}
        if "__type__" in obj:
            try:
                ctx.raise_if_nodecode(obj["__type__"])
                base = obj["__type__"].split(".")[0]
                if base == "user":
                    name = ".".join(obj["__type__"].split(".")[1:])
                    if decoder := user.decoders.get(name):
                        obj = decoder(obj, ctx)
                else:
                    obj = get_decoders()[base](obj, ctx)
            except TypeIsNodecode:
                pass
    elif isinstance(obj, list):
        return [_from_jsonable(v, ctx / str(i)) for i, v in enumerate(obj)]
    elif isinstance(obj, tuple):
        return tuple(
            _from_jsonable(v, ctx / str(i)) for i, v in enumerate(obj)
        )
    return obj


def _make_or_set_ctx(
    file_path: str | Path | None, ctx: Context | None, **kwargs
) -> Context:
    """
    Generate a context object that is consistent with the inputs.
    """
    if file_path is None and ctx is None:
        raise ValueError(
            "Either a file path or a context (or both) must be provided."
        )
    if file_path is not None and ctx is not None:
        if ctx.file_path is not None and ctx.file_path != file_path:
            raise ValueError(
                "The file path in the context does not match the provided "
                "file path."
            )
        assert isinstance(file_path, (str, Path))  # for typechecking
        ctx.file_path = Path(file_path)
    if ctx is None:
        ctx = Context(file_path=file_path, **kwargs)
    return ctx


def _to_jsonable(obj: Any, ctx: Context) -> Any:
    """
    Transforms an object (dict, list, primitive) that possibly contains types
    that TurboBroccoli's custom encoders support, and returns an object that is
    readily vanilla JSON-serializable.
    """
    name = obj.__class__.__name__
    if name in user.encoders:
        obj = user.encoders[name](obj, ctx)
    for encoder in get_encoders():
        try:
            obj = encoder(obj, ctx)
            break
        except TypeNotSupported:
            pass
    if isinstance(obj, dict):
        return {k: _to_jsonable(v, ctx / k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v, ctx / str(i)) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(_to_jsonable(v, ctx / str(i)) for i, v in enumerate(obj))
    return obj


def from_json(doc: str, ctx: Context | None = None) -> Any:
    """
    Deserializes a JSON string. The context's file path and compression setting
    will be ignored.
    """
    return _from_jsonable(json.loads(doc), Context() if ctx is None else ctx)


def load_json(
    file_path: str | Path | None = None, ctx: Context | None = None, **kwargs
) -> Any:
    """
    Loads a JSON file.

    Args:
        file_path (str | Path | None): If left to `None`, a context with a file
            path must be provided
        ctx (Context | None): The context to use. If `None`, a new context will
            be created with the kwargs.
        **kwargs: Forwarded to the `turbo_broccoli.context.Context`
            constructor. If `ctx` is provided, the kwargs are ignored.
    """
    ctx = _make_or_set_ctx(file_path, ctx, **kwargs)
    assert isinstance(ctx.file_path, Path)  # for typechecking
    if ctx.compress:
        with ctx.file_path.open(mode="rb") as fp:
            s = zlib.decompress(fp.read()).decode("utf-8")
        return _from_jsonable(json.loads(s), ctx)
    with ctx.file_path.open(mode="r", encoding="utf-8") as fp:
        return _from_jsonable(json.load(fp), ctx)


def save_json(
    obj: Any,
    file_path: str | Path | None = None,
    ctx: Context | None = None,
    **kwargs,
) -> None:
    """
    Serializes an object and writes the result to a file. The artifact path and
    the output file's parent folder will be created if they don't exist.

    Args:
        obj (Any):
        file_path (str | Path):
        ctx (Context | None): The context to use. If `None`, a new context will
            be created with the kwargs.
        **kwargs: Forwarded to the `turbo_broccoli.context.Context`
            constructor.
    """
    ctx = _make_or_set_ctx(file_path, ctx, **kwargs)
    data = json.dumps(_to_jsonable(obj, ctx))
    assert isinstance(ctx.file_path, Path)  # for typechecking
    if not ctx.file_path.parent.exists():
        ctx.file_path.parent.mkdir(parents=True)
    if ctx.compress:
        with ctx.file_path.open(mode="wb") as fp:
            fp.write(zlib.compress(data.encode("utf-8")))
    else:
        with ctx.file_path.open(mode="w", encoding="utf-8") as fp:
            fp.write(data)


def to_json(obj: Any, ctx: Context | None = None) -> str:
    """
    Converts an object to a JSON string. The context's artifact folder will be
    created if it doesn't exist. The context's file path and compression
    setting will be ignored.
    """
    ctx = Context() if ctx is None else ctx
    if not ctx.artifact_path.exists():
        ctx.artifact_path.mkdir(parents=True)
    return json.dumps(_to_jsonable(obj, ctx))
