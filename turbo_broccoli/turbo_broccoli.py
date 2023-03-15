"""Main module containing the JSON encoder and decoder classes."""
__docformat__ = "google"

import json
from typing import Any, Callable, Dict, List


import turbo_broccoli.bytes
import turbo_broccoli.collections
import turbo_broccoli.dataclass
import turbo_broccoli.generic

try:
    import turbo_broccoli.keras

    HAS_KERAS = True
except ModuleNotFoundError:
    HAS_KERAS = False

try:
    import turbo_broccoli.numpy

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False

try:
    import turbo_broccoli.pandas

    HAS_PANDAS = True
except ModuleNotFoundError:
    HAS_PANDAS = False


try:
    import turbo_broccoli.secret

    HAS_SECRET = True
except ModuleNotFoundError:
    HAS_SECRET = False

try:
    import turbo_broccoli.tensorflow

    HAS_TENSORFLOW = True
except ModuleNotFoundError:
    HAS_TENSORFLOW = False

try:
    import turbo_broccoli.pytorch

    HAS_PYTORCH = True
except ModuleNotFoundError:
    HAS_PYTORCH = False


try:
    import turbo_broccoli.scipy

    HAS_SCIPY = True
except ModuleNotFoundError:
    HAS_SCIPY = False

try:
    import turbo_broccoli.sklearn

    HAS_SKLEARN = True
except ModuleNotFoundError:
    HAS_SKLEARN = False


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
            "__bytes__": turbo_broccoli.bytes.from_json,
            "__collections__": turbo_broccoli.collections.from_json,
            "__dataclass__": turbo_broccoli.dataclass.from_json,
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
        for t, f in DECODERS.items():
            if t in dct:
                return f(dct)
        return dct


class TurboBroccoliEncoder(json.JSONEncoder):
    """
    TurboBroccoli's custom JSON decoder class. See the README for the list of
    supported types.
    """

    def default(self, o: Any) -> Any:

        ENCODERS: List[Callable[[Any], dict]] = [
            turbo_broccoli.bytes.to_json,
            turbo_broccoli.collections.to_json,
            turbo_broccoli.dataclass.to_json,
            turbo_broccoli.generic.to_json,
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
        for f in ENCODERS:
            try:
                return f(o)
            except TypeError:
                pass
        return super().default(o)

    def encode(self, o: Any) -> str:
        """
        Reimplementation of encode just to treat the `namedtuple` case. An
        object is considered a namedtuple if it is an instance of `tuple` and
        has the following attributes: `_asdict`, `_field_defaults`, `_fields`,
        `_make`, `_replace`. In this case,
        `turbo_broccoli.collections._namedtuple_to_json` is called directly.
        """
        attrs = ["_asdict", "_field_defaults", "_fields", "_make", "_replace"]
        if isinstance(o, tuple) and all(map(lambda a: hasattr(o, a), attrs)):
            d = turbo_broccoli.collections._namedtuple_to_json(o)
            return super().encode({"__collections__": d})
        return super().encode(o)


def from_json(doc: str) -> Any:
    """Converts a JSON document back to a Python object."""
    return json.loads(doc, cls=TurboBroccoliDecoder)


def to_json(obj: Any) -> str:
    """Converts an object to JSON."""
    return json.dumps(obj, cls=TurboBroccoliEncoder)
