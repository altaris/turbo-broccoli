# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Decode exclusion tests"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from common import assert_to_from_json, to_from_json
from test_pandas import _assert_equal as assert_equal_pd
from test_pytorch import TestModule

from turbo_broccoli import Context


def _basic_dict() -> dict:
    return {"a_list": [1, "2", None], "a_str": "abcd", "an_int": 42}


def test_nodecode_nothing():
    assert_to_from_json(_basic_dict())


def test_nodecode_bytes():
    ctx = Context(nodecode_types=["bytes"])
    x = {"b": "Hello 🌎".encode("utf8"), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    for k in _basic_dict():
        assert x[k] == y[k]
    assert set(y["b"].keys()) == {"__type__", "__version__", "data"}
    assert y["b"]["__type__"] == "bytes"


def test_nodecode_numpy_artefact():
    ctx = Context(min_artifact_size=0, nodecode_types=["numpy"])
    x = {"b": np.random.random((100, 100)), **_basic_dict()}
    # assert_equal_np(x, to_from_json(x, ctx))
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["b"].keys()) == {"__type__", "__version__", "data"}
    assert y["b"]["__type__"] == "numpy.ndarray"
    assert isinstance(y["b"]["data"], bytes)


def test_nodecode_bytes_numpy():
    ctx = Context(min_artifact_size=0, nodecode_types=["bytes"])
    x = {"b": np.random.random((100, 100)), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["b"].keys()) == {"__type__", "__version__", "data"}
    assert y["b"]["__type__"] == "numpy.ndarray"
    assert isinstance(y["b"]["data"], dict)
    assert set(y["b"]["data"].keys()) == {"__type__", "__version__", "id"}
    assert y["b"]["data"]["__type__"] == "bytes"


def test_nodecode_bytes_tensorflow():
    ctx = Context(min_artifact_size=0, nodecode_types=["bytes"])
    x = {"b": tf.random.uniform((100, 100), dtype=tf.float64), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["b"].keys()) == {"__type__", "__version__", "data"}
    assert y["b"]["__type__"] == "tensorflow.tensor"
    assert isinstance(y["b"]["data"], dict)
    assert set(y["b"]["data"].keys()) == {"__type__", "__version__", "id"}
    assert y["b"]["data"]["__type__"] == "bytes"


def test_nodecode_bytes_torch_tensor():
    ctx = Context(min_artifact_size=0, nodecode_types=["bytes"])
    x = {"b": torch.rand((100, 100)), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["b"].keys()) == {"__type__", "__version__", "data"}
    assert y["b"]["__type__"] == "pytorch.tensor"
    assert isinstance(y["b"]["data"], dict)
    assert set(y["b"]["data"].keys()) == {"__type__", "__version__", "id"}
    assert y["b"]["data"]["__type__"] == "bytes"


def test_nodecode_bytes_torch_module():
    ctx = Context(min_artifact_size=0, nodecode_types=["bytes"])
    x = {"b": TestModule(), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["b"].keys()) == {"__type__", "__version__", "state"}
    assert y["b"]["__type__"] == "pytorch.module.TestModule"
    assert isinstance(y["b"]["state"], dict)
    assert set(y["b"]["state"].keys()) == {"__type__", "__version__", "id"}
    assert y["b"]["state"]["__type__"] == "bytes"


def test_nodecode_dataclass():
    @dataclass
    class C:
        a_byte_str: bytes
        a_list: list
        a_str: str
        an_int: int

    @dataclass
    class D:
        a_dataclass: C
        a_float: float

    ctx = Context(nodecode_types=["dataclass.C"], dataclass_types=[C, D])
    c = C(a_byte_str=b"", a_list=[], a_str="", an_int=0)
    x = {"c": c, "d": D(a_dataclass=c, a_float=1.2), **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["c"].keys()) == {"__type__", "__version__", "data"}
    assert y["c"]["__type__"] == "dataclass.C"
    assert isinstance(y["d"], D)
    assert isinstance(y["d"].a_dataclass, dict)
    assert y["d"].a_float == 1.2


def test_nodecode_pandas_series():
    ctx = Context(nodecode_types=["pandas.series"])
    s = pd.Series([1, 2, 3])
    x = {"s": s, **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["s"].keys()) == {"__type__", "__version__", "data", "name"}
    assert y["s"]["__type__"] == "pandas.series"


def test_nodecode_pandas_dataframe():
    ctx = Context(nodecode_types=["pandas.dataframe"])
    s = pd.Series([1, 2, 3])
    d = pd.DataFrame({"a": s, "b": pd.Categorical(["X", "Y", "X"])})
    x = {"s": s, "d": d, **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["s"].keys()) == {"__type__", "__version__", "data", "name"}
    assert set(y["d"].keys()) == {"__type__", "__version__", "data", "dtypes"}
    assert y["s"]["__type__"] == "pandas.series"
    assert y["d"]["__type__"] == "pandas.dataframe"


def test_nodecode_pandas_series_dataframe():
    ctx = Context(nodecode_types=["pandas.series"])
    s = pd.Series([1, 2, 3])
    d = pd.DataFrame({"a": s, "b": pd.Categorical(["X", "Y", "X"])})
    x = {"s": s, "d": d, **_basic_dict()}
    y = to_from_json(x, ctx)
    assert set(x.keys()) == set(y.keys())
    assert set(y["s"].keys()) == {"__type__", "__version__", "data", "name"}
    assert y["s"]["__type__"] == "pandas.series"
    assert isinstance(y["d"], pd.DataFrame)
    assert_equal_pd(x["d"], y["d"])
