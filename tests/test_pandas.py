# pylint: disable=missing-function-docstring
"""Numpy (de)serialization test suite"""

from datetime import datetime

import pandas as pd
from common import to_from_json


def _assert_equal(a, b):
    assert type(a) == type(b)  # pylint: disable=unidiomatic-typecheck
    if isinstance(a, pd.DataFrame):
        assert ((a == b).all()).all()
    elif isinstance(a, pd.Series):
        assert (a == b).all()


def test_series_int():
    x = pd.Series(range(10))
    _assert_equal(x, to_from_json(x))
    x = pd.Series(range(10), name="a_series")
    _assert_equal(x, to_from_json(x))


def test_series_float():
    x = pd.Series([1.0, 0.1, 1e3], name="a_series")
    _assert_equal(x, to_from_json(x))


# def test_series_complex():
#     x = pd.Series([1.0, 0.1j, 1e3 + 1e4j], name="a_series")
#     _assert_equal(x, to_from_json(x))


def test_series_string():
    x = pd.Series(["a", "b", "c", "🧇"], name="a_series")
    _assert_equal(x, to_from_json(x))


def test_series_datetime():
    x = pd.Series(
        ["2013-01-01", "2013-01-02", "2013-01-03", datetime.now()],
        dtype="datetime64[ns]",
        name="a_series",
    )
    _assert_equal(x, to_from_json(x))


def test_series_categorical():
    x = pd.Series(
        ["A", "B", "A", "C", "B", "A"],
        dtype="category",
        name="a_series",
    )
    _assert_equal(x, to_from_json(x))


def test_dataframe():
    x = pd.DataFrame(
        {
            "float": pd.Series(["1", ".1", "1e-10", "1e11"], dtype="float64"),
            "date": datetime.now(),
            # "td": pd.Timedelta("1 day"),
            "string": "Hello world 👋",
            # "cplx": [1, 2 + 3j, 3.14, .0000001],
            "cat": pd.Categorical(["X", "Y", "X", "Z"]),
        },
    )
    _assert_equal(x, to_from_json(x))
