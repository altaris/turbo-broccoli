# pylint: disable=missing-function-docstring
"""Numpy (de)serialization test suite"""

import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from common import from_json, to_json


def _assert_equal(a, b):
    assert isinstance(a, (np.ndarray, np.number))
    assert_equal(type(a), type(b))
    assert_equal(a.dtype, b.dtype)
    if isinstance(a, np.ndarray):
        assert_array_equal(a, b)
    else:
        assert_equal(a, b)


def test_ndarray():
    x = np.array([])
    assert_equal(x, from_json(to_json(x)))
    x = np.array(1, dtype="int8")
    assert_equal(x, from_json(to_json(x)))
    x = np.array(1, dtype="float32")
    assert_equal(x, from_json(to_json(x)))
    x = np.random.random((1, 2, 3, 4, 5))
    assert_equal(x, from_json(to_json(x)))


def test_number():
    x = np.float32(-1.2)
    _assert_equal(x, from_json(to_json(x)))
    x = np.float32(0.0000000001)
    _assert_equal(x, from_json(to_json(x)))
    x = np.float32(1e10)
    _assert_equal(x, from_json(to_json(x)))
    x = np.int64(420)
    _assert_equal(x, from_json(to_json(x)))
