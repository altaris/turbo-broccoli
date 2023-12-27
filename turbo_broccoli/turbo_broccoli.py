# pylint: disable=bare-except
"""Main module containing the JSON encoder and decoder classes."""

import json
from pathlib import Path
from typing import Any, Callable


import turbo_broccoli.bytes
import turbo_broccoli.collections
import turbo_broccoli.dataclass
from turbo_broccoli.environment import get_artifact_path, set_artifact_path
import turbo_broccoli.generic
import turbo_broccoli.dict
from turbo_broccoli.utils import TypeIsNodecode, TypeNotSupported

try:
    import turbo_broccoli.keras

    HAS_KERAS = True
except:
    HAS_KERAS = False

try:
    import turbo_broccoli.numpy

    HAS_NUMPY = True
except:
    HAS_NUMPY = False

try:
    import turbo_broccoli.pandas

    HAS_PANDAS = True
except:
    HAS_PANDAS = False


try:
    import turbo_broccoli.secret

    HAS_SECRET = True
except:
    HAS_SECRET = False

try:
    import turbo_broccoli.tensorflow

    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

try:
    import turbo_broccoli.pytorch

    HAS_PYTORCH = True
except:
    HAS_PYTORCH = False


try:
    import turbo_broccoli.scipy

    HAS_SCIPY = True
except:
    HAS_SCIPY = False

try:
    import turbo_broccoli.sklearn

    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False


try:
    import turbo_broccoli.bokeh

    HAS_BOKEH = True
except:
    HAS_BOKEH = False


class TurboBroccoliDecoder(json.JSONDecoder):
    """
    TurboBroccoli's custom JSON decoder class. See the README for the list of
    supported types.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self._hook, *args, **kwargs)

    def _hook(self, dct):
        """Deserialization hook"""
        DECODERS: dict[str, Callable[[dict], Any]] = {
            "dict": turbo_broccoli.dict.from_json,
            "bytes": turbo_broccoli.bytes.from_json,
        }
        if HAS_KERAS:
            DECODERS["keras"] = turbo_broccoli.keras.from_json
        if HAS_NUMPY:
            DECODERS["numpy"] = turbo_broccoli.numpy.from_json
        if HAS_PANDAS:
            DECODERS["pandas"] = turbo_broccoli.pandas.from_json
        if HAS_PYTORCH:
            DECODERS["pytorch"] = turbo_broccoli.pytorch.from_json
        if HAS_SECRET:
            DECODERS["secret"] = turbo_broccoli.secret.from_json
        if HAS_TENSORFLOW:
            DECODERS["tensorflow"] = turbo_broccoli.tensorflow.from_json
        if HAS_SCIPY:
            DECODERS["scipy"] = turbo_broccoli.scipy.from_json
        if HAS_SKLEARN:
            DECODERS["sklearn"] = turbo_broccoli.sklearn.from_json
        if HAS_BOKEH:
            DECODERS["bokeh"] = turbo_broccoli.bokeh.from_json
        # Intentionally put last
        DECODERS["collections"] = turbo_broccoli.collections.from_json
        DECODERS["dataclass"] = turbo_broccoli.dataclass.from_json
        for t, f in DECODERS.items():
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
        ENCODERS: list[Callable[[Any], dict]] = [
            turbo_broccoli.bytes.to_json,
        ]
        if HAS_KERAS:
            ENCODERS.append(turbo_broccoli.keras.to_json)
        if HAS_NUMPY:
            ENCODERS.append(turbo_broccoli.numpy.to_json)
        if HAS_PANDAS:
            ENCODERS.append(turbo_broccoli.pandas.to_json)
        if HAS_PYTORCH:
            ENCODERS.append(turbo_broccoli.pytorch.to_json)
        if HAS_SECRET:
            ENCODERS.append(turbo_broccoli.secret.to_json)
        if HAS_TENSORFLOW:
            ENCODERS.append(turbo_broccoli.tensorflow.to_json)
        if HAS_SCIPY:
            ENCODERS.append(turbo_broccoli.scipy.to_json)
        if HAS_SKLEARN:
            ENCODERS.append(turbo_broccoli.sklearn.to_json)
        if HAS_BOKEH:
            ENCODERS.append(turbo_broccoli.bokeh.to_json)
        # Intentionally put last
        ENCODERS += [
            turbo_broccoli.collections.to_json,
            turbo_broccoli.dataclass.to_json,
            turbo_broccoli.generic.to_json,
        ]
        for f in ENCODERS:
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
        PRIORITY_ENCODERS: list[Callable[[Any], dict]] = [
            turbo_broccoli.dict.to_json,
            turbo_broccoli.collections.to_json,
        ]
        for f in PRIORITY_ENCODERS:
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
