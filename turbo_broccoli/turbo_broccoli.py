# pylint: disable=bare-except
"""Main module containing the JSON encoder and decoder classes."""
__docformat__ = "google"

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Union


import turbo_broccoli.bytes
import turbo_broccoli.collections
import turbo_broccoli.dataclass
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
        DECODERS: Dict[str, Callable[[dict], Any]] = {
            "__dict__": turbo_broccoli.dict.from_json,
            "__bytes__": turbo_broccoli.bytes.from_json,
        }
        if HAS_KERAS:
            DECODERS["__keras__"] = turbo_broccoli.keras.from_json
        if HAS_NUMPY:
            DECODERS["__numpy__"] = turbo_broccoli.numpy.from_json
        if HAS_PANDAS:
            DECODERS["__pandas__"] = turbo_broccoli.pandas.from_json
        if HAS_PYTORCH:
            DECODERS["__pytorch__"] = turbo_broccoli.pytorch.from_json
        if HAS_SECRET:
            DECODERS["__secret__"] = turbo_broccoli.secret.from_json
        if HAS_TENSORFLOW:
            DECODERS["__tensorflow__"] = turbo_broccoli.tensorflow.from_json
        if HAS_SCIPY:
            DECODERS["__scipy__"] = turbo_broccoli.scipy.from_json
        if HAS_SKLEARN:
            DECODERS["__sklearn__"] = turbo_broccoli.sklearn.from_json
        if HAS_BOKEH:
            DECODERS["__bokeh__"] = turbo_broccoli.bokeh.from_json
        # Intentionally put last
        DECODERS["__collections__"] = turbo_broccoli.collections.from_json
        DECODERS["__dataclass__"] = turbo_broccoli.dataclass.from_json
        for t, f in DECODERS.items():
            if t in dct:
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

        ENCODERS: List[Callable[[Any], dict]] = [
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
        PRIORITY_ENCODERS: List[Callable[[Any], dict]] = [
            turbo_broccoli.dict.to_json,
            turbo_broccoli.collections.to_json,
        ]
        for f in PRIORITY_ENCODERS:
            try:
                return super().encode(f(o))
            except TypeNotSupported:
                pass
        return super().encode(o)


def from_json(doc: str) -> Any:
    """Converts a JSON document string back to a Python object"""
    return json.loads(doc, cls=TurboBroccoliDecoder)


def load_json(path: Union[str, Path]) -> Any:
    """Loads and deserializes a JSON file using Turbo Broccoli"""
    with open(path, mode="r", encoding="utf-8") as fp:
        return json.load(fp, cls=TurboBroccoliDecoder)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Serializes and saves a JSON-serializable object"""
    with open(path, mode="w", encoding="utf-8") as fp:
        fp.write(to_json(obj))


def to_json(obj: Any) -> str:
    """Converts an object to JSON"""
    return json.dumps(obj, cls=TurboBroccoliEncoder)
