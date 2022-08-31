# pylint: disable=missing-function-docstring
"""Decode exclusion tests"""

from collections import deque

import numpy as np
import pandas as pd
import tensorflow as tf
from turbo_broccoli.environment import set_nodecode

from common import from_json, to_json
from test_dataclass import C, D
from test_keras import _build_model
from test_pandas import _assert_equal as assert_equal_pd


def _basic_dict() -> dict:
    return {"a_list": [1, "2", None], "a_str": "abcd", "an_int": 42}


def test_nodecode_nothing():
    set_nodecode("")
    x = _basic_dict()
    assert x == from_json(to_json(x))
    set_nodecode("")


def test_nodecode_bytes():
    set_nodecode("bytes")
    x = {"b": "Hello 🌎".encode("utf8"), **_basic_dict()}
    y = {"b": None, **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")


def test_nodecode_dataclass():
    set_nodecode("dataclass.C")
    c = C(a_byte_str=b"", a_list=[], a_str="", an_int=0)
    x = {"dcc": c, "dcd": D(a_dataclass=c, a_float=1.2), **_basic_dict()}
    y = {"dcc": None, "dcd": D(a_dataclass=None, a_float=1.2), **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")


def test_nodecode_collections():
    set_nodecode("collections.deque")
    x = {"deq": deque(range(100)), **_basic_dict()}
    y = {"deq": None, **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")


def test_nodecode_keras():
    set_nodecode("keras.model")
    x = {"model": _build_model(), **_basic_dict()}
    y = {"model": None, **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")


def test_nodecode_numpy():
    set_nodecode("numpy.ndarray")
    x = {"arr": np.zeros(10), **_basic_dict()}
    y = {"arr": None, **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")


# The serialization of Series depends on that of DataFrame. So excluding
# pandas.dataframe will make it impossible to deserialize Series.
# def test_nodecode_pandas_df():
#     set_nodecode("pandas.dataframe")
#     s = pd.Series([1, 2, 3])
#     df = pd.DataFrame(
#         {
#             "a": s,
#             "b": pd.Categorical(["X", "Y", "X"])
#         }
#     )
#     x = {
#         "s": s,
#         "df": df,
#     }
#     print(to_json(x))
#     y = from_json(to_json(x))
#     assert_equal_pd(x["s"], y["s"])
#     assert y["df"] is None
#     set_nodecode("")


def test_nodecode_pandas_ser():
    set_nodecode("pandas.series")
    s = pd.Series([1, 2, 3])
    df = pd.DataFrame({"a": s, "b": pd.Categorical(["X", "Y", "X"])})
    x = {
        "s": s,
        "df": df,
    }
    y = from_json(to_json(x))
    assert_equal_pd(x["df"], y["df"])
    assert y["s"] is None
    set_nodecode("")


def test_nodecode_tensorflow():
    set_nodecode("tensorflow.tensor")
    x = {"t": tf.random.uniform((10, 10)), **_basic_dict()}
    y = {"t": None, **_basic_dict()}
    assert y == from_json(to_json(x))
    set_nodecode("")
