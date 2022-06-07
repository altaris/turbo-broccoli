"""Main module containing the JSON encoder and decoder classes."""
__docformat__ = "google"

import json
from typing import Any, Callable, Dict, List


import turbo_broccoli.numpy


class TurboBroccoliDecoder(json.JSONDecoder):
    """
    TurboBroccoli's custom JSON decoder class. See the README for the list of
    supported types.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(object_hook=self._hook, *args, **kwargs)

    def _hook(self, dct):
        """Deserialization hook"""
        DECODERS: Dict[str, Callable[[dict], Any]] = {}
        if turbo_broccoli.numpy.HAS_NUMPY:
            DECODERS["__numpy__"] = turbo_broccoli.numpy.from_json
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

        ENCODERS: List[Callable[[Any], dict]] = []
        if turbo_broccoli.numpy.HAS_NUMPY:
            ENCODERS.append(turbo_broccoli.numpy.to_json)
        for f in ENCODERS:
            try:
                return f(o)
            except TypeError:
                pass
        return super().default(o)
