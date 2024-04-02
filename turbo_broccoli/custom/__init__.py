# pylint: disable=bare-except
"""Custom type encoder and decoders, all grouped in a dedicated submodule"""

from typing import Any, Callable

from turbo_broccoli.context import Context
from turbo_broccoli.custom import bytes as _bytes
from turbo_broccoli.custom import collections as _collections
from turbo_broccoli.custom import dataclass as _dataclass
from turbo_broccoli.custom import datetime as _datetime
from turbo_broccoli.custom import dct as _dict
from turbo_broccoli.custom import embedded as _embedded
from turbo_broccoli.custom import generic as _generic
from turbo_broccoli.custom import uuid as _uuid

try:
    from turbo_broccoli.custom import keras as _keras

    HAS_KERAS = True
except:
    HAS_KERAS = False

try:
    from turbo_broccoli.custom import numpy as _numpy

    HAS_NUMPY = True
except:
    HAS_NUMPY = False

try:
    from turbo_broccoli.custom import pandas as _pandas

    HAS_PANDAS = True
except:
    HAS_PANDAS = False


try:
    from turbo_broccoli.custom import secret as _secret

    HAS_SECRET = True
except:
    HAS_SECRET = False

try:
    from turbo_broccoli.custom import tensorflow as _tensorflow

    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

try:
    from turbo_broccoli.custom import pytorch as _pytorch

    HAS_PYTORCH = True
except:
    HAS_PYTORCH = False

try:
    from turbo_broccoli.custom import scipy as _scipy

    HAS_SCIPY = True
except:
    HAS_SCIPY = False

try:
    from turbo_broccoli.custom import sklearn as _sklearn

    HAS_SKLEARN = True
except:
    HAS_SKLEARN = False

try:
    from turbo_broccoli.custom import bokeh as _bokeh

    HAS_BOKEH = True
except:
    HAS_BOKEH = False


def get_decoders() -> dict[str, Callable[[dict, Context], Any]]:
    """
    Returns the dict of all available decoders, which looks like this:

    ```py
    {
        "mytype": mytype_decoder,
        ...
    }
    ```

    `mytype_decoder` is a function that takes an vanilla JSON dict that
    looks like this (excluding comments):

    ```py
    {
        "__type__": "mytype.mysubtype",  # or simply "mytype"
        "__version__": <int>,
        ...
    }
    ```
    """
    decoders: dict[str, Callable[[dict, Context], Any]] = {
        "bytes": _bytes.from_json,
        "datetime": _datetime.from_json,
        "dict": _dict.from_json,
        "uuid": _uuid.from_json,
    }
    if HAS_KERAS:
        decoders["keras"] = _keras.from_json
    if HAS_NUMPY:
        decoders["numpy"] = _numpy.from_json
    if HAS_PANDAS:
        decoders["pandas"] = _pandas.from_json
    if HAS_PYTORCH:
        decoders["pytorch"] = _pytorch.from_json
    if HAS_SECRET:
        decoders["secret"] = _secret.from_json
    if HAS_TENSORFLOW:
        decoders["tensorflow"] = _tensorflow.from_json
    if HAS_SCIPY:
        decoders["scipy"] = _scipy.from_json
    if HAS_SKLEARN:
        decoders["sklearn"] = _sklearn.from_json
    if HAS_BOKEH:
        decoders["bokeh"] = _bokeh.from_json
    # Intentionally put last
    decoders["collections"] = _collections.from_json
    decoders["dataclass"] = _dataclass.from_json
    decoders["embedded"] = _embedded.from_json
    return decoders


def get_encoders() -> list[Callable[[Any, Context], dict]]:
    """
    Returns the dict of all available encoder. An encoder is a function that
    takes an object and returns a readily vanilla JSON-serializable dict. This
    this should be of the form

    ```py
    {
        "__type__": "mytype.mysubtype",  # or simply "mytype"
        "__version__": <int>,
        ...
    }
    ```

    The encoder should raise a `turbo_broccoli.utils.TypeNotSupported` if it
    doesn't handle the kind of object it was given.
    """
    encoders: list[Callable[[Any, Context], dict]] = [
        _bytes.to_json,
        _datetime.to_json,
        _dict.to_json,
        _uuid.to_json,
    ]
    if HAS_KERAS:
        encoders.append(_keras.to_json)
    if HAS_NUMPY:
        encoders.append(_numpy.to_json)
    if HAS_PANDAS:
        encoders.append(_pandas.to_json)
    if HAS_PYTORCH:
        encoders.append(_pytorch.to_json)
    if HAS_SECRET:
        encoders.append(_secret.to_json)
    if HAS_TENSORFLOW:
        encoders.append(_tensorflow.to_json)
    if HAS_SCIPY:
        encoders.append(_scipy.to_json)
    if HAS_SKLEARN:
        encoders.append(_sklearn.to_json)
    if HAS_BOKEH:
        encoders.append(_bokeh.to_json)
    # Intentionally put last
    encoders += [
        _collections.to_json,
        _dataclass.to_json,
        _generic.to_json,
        _embedded.to_json,
    ]
    return encoders
