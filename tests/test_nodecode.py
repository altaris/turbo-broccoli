# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""Decode exclusion tests"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from common import assert_to_from_json, to_from_json
from test_keras import _build_model
from test_pandas import _assert_equal as assert_equal_pd

from turbo_broccoli import Context


def _basic_dict() -> dict:
    return {"a_list": [1, "2", None], "a_str": "abcd", "an_int": 42}


def test_nodecode_nothing():
    assert_to_from_json(_basic_dict())


def test_nodecode_bytes():
    ctx = Context(nodecode_types=["bytes"])
    x = {"b": "Hello 🌎".encode("utf8"), **_basic_dict()}
    y = {"b": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)


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
    y = {"c": None, "d": D(a_dataclass=None, a_float=1.2), **_basic_dict()}
    assert y == to_from_json(x, ctx)


def test_nodecode_collections():
    ctx = Context(nodecode_types=["collections.deque"])
    x = {"deq": deque(range(100)), **_basic_dict()}
    y = {"deq": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)


def test_nodecode_keras():
    ctx = Context(nodecode_types=["keras.model"])
    x = {"model": _build_model(), **_basic_dict()}
    y = {"model": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)


def test_nodecode_numpy():
    ctx = Context(nodecode_types=["numpy.ndarray"])
    x = {"arr": np.zeros(10), **_basic_dict()}
    y = {"arr": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)


def test_nodecode_pandas_series():
    ctx = Context(nodecode_types=["pandas.series"])
    s = pd.Series([1, 2, 3])
    x = {"ser": s, **_basic_dict()}
    y = {"ser": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)


def test_nodecode_pandas_dataframe():
    ctx = Context(nodecode_types=["pandas.dataframe"])
    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": s, "b": pd.Categorical(["X", "Y", "X"])})
    x = {"s": s, "df": df, **_basic_dict()}
    y = {"s": None, "df": None, **_basic_dict()}
    y = to_from_json(x, ctx)
    assert y == to_from_json(x, ctx)


def test_nodecode_pandas_ser():
    ctx = Context(nodecode_types=["pandas.series"])
    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": s, "b": pd.Categorical(["X", "Y", "X"])})
    x = {
        "s": s,
        "df": df,
    }
    y = to_from_json(x, ctx)
    assert_equal_pd(x["df"], y["df"])
    assert y["s"] is None


def test_nodecode_tensorflow():
    ctx = Context(nodecode_types=["tensorflow.tensor"])
    x = {"t": tf.random.uniform((10, 10)), **_basic_dict()}
    y = {"t": None, **_basic_dict()}
    assert y == to_from_json(x, ctx)
