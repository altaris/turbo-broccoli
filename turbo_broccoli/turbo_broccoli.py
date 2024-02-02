"""Main module containing the JSON encoder and decoder classes."""

import json
from pathlib import Path
from typing import Any, Callable

from turbo_broccoli.environment import get_artifact_path, set_artifact_path
from turbo_broccoli.utils import TypeIsNodecode, TypeNotSupported
from turbo_broccoli.custom import (
    _collections,
    get_decoders,
    get_encoders,
)


class TurboBroccoliDecoder(json.JSONDecoder):
    """
    TurboBroccoli's custom JSON decoder class. See the README for the list of
    supported types.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self._hook, *args, **kwargs)

    def _hook(self, dct):
        """Deserialization hook"""
        for t, f in get_decoders().items():
            if str(dct.get("__type__", "")).startswith(t):
                try:
                    return f(dct)
                except TypeIsNodecode:
                    return None
        return dct


class TurboBroccoliEncoder(json.JSONEncoder):
    """
    TurboBroccoli's custom JSON decoder class. See the README for the list of
    supported types.
    """

    def default(self, o: Any) -> Any:
        for f in get_encoders():
            try:
                return f(o)
            except TypeNotSupported:
                pass
        return super().default(o)

    def encode(self, o: Any) -> str:
        """
        Reimplementation of encode just to treat exceptional cases that need to
        be handled before `JSONEncoder.encode`.
        """
        priority_encoders: list[Callable[[Any], dict]] = [
            _collections.to_json,
        ]
        for f in priority_encoders:
            try:
                return super().encode(f(o))
            except TypeNotSupported:
                pass
        return super().encode(o)


# pylint: disable=missing-function-docstring
def from_json(doc: str) -> Any:
    """Converts a JSON document string back to a Python object"""
    return json.loads(doc, cls=TurboBroccoliDecoder)


def load_json(path: str | Path, auto_artifact_path: bool = True) -> Any:
    """
    Loads and deserializes a JSON file using Turbo Broccoli

    Args:
        path (str | Path):
        auto_artifact_path (bool): If left to `True`, set the artifact path to
            the target file's parent directory before loading. After loading,
            the previous artifact path is restored. See also see
            `turbo_broccoli.environment.set_artifact_path`.
    """
    old_artifact_path = get_artifact_path()
    if auto_artifact_path:
        set_artifact_path(Path(path).parent)
    with open(path, mode="r", encoding="utf-8") as fp:
        document = json.load(fp, cls=TurboBroccoliDecoder)
    if auto_artifact_path:
        set_artifact_path(old_artifact_path)
    return document


def save_json(
    obj: Any, path: str | Path, auto_artifact_path: bool = True
) -> None:
    """
    Serializes and saves a JSON-serializable object

    Args:
        obj (Any):
        path (str | Path):
        auto_artifact_path (bool): If left to `True`, set the artifact path to
            the target file's parent directory before saving. After saving,
            the previous artifact path is restored. See also see
            `turbo_broccoli.environment.set_artifact_path`.
    """
    old_artifact_path = get_artifact_path()
    if auto_artifact_path:
        set_artifact_path(Path(path).parent)
    with open(path, mode="w", encoding="utf-8") as fp:
        fp.write(to_json(obj))
    if auto_artifact_path:
        set_artifact_path(old_artifact_path)


def to_json(obj: Any) -> str:
    """Converts an object to JSON"""
    return json.dumps(obj, cls=TurboBroccoliEncoder)
