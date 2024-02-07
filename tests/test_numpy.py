# pylint: disable=missing-function-docstring
"""Numpy (de)serialization test suite"""

import numpy as np
from common import assert_to_from_json, to_from_json
from numpy.testing import assert_array_equal, assert_equal

from turbo_broccoli import Context


def _assert_equal(a, b):
    assert isinstance(a, (np.ndarray, np.number))
    assert_equal(type(a), type(b))
    assert_equal(a.dtype, b.dtype)
    if isinstance(a, np.ndarray):
        assert_array_equal(a, b)
    else:
        assert_equal(a, b)


def test_numpy_number():
    x = np.float32(-1.2)
    _assert_equal(x, to_from_json(x))
    x = np.float32(0.0000000001)
    _assert_equal(x, to_from_json(x))
    x = np.float32(1e10)
    _assert_equal(x, to_from_json(x))
    x = np.int64(420)
    _assert_equal(x, to_from_json(x))


def test_numpy_array():
    x = np.array([])
    _assert_equal(x, to_from_json(x))
    x = np.array(1, dtype="int8")
    _assert_equal(x, to_from_json(x))
    x = np.array(1, dtype="float32")
    _assert_equal(x, to_from_json(x))
    x = np.random.random((1, 2, 3, 4, 5))
    _assert_equal(x, to_from_json(x))


def test_numpy_large_array():
    ctx = Context(min_artifact_size=0)
    x = np.random.random((100, 100))
    _assert_equal(x, to_from_json(x, ctx))


def test_numpy_dtype():
    assert_to_from_json(np.dtype(np.float64))
    assert_to_from_json(np.dtype(int))
    assert_to_from_json(np.dtype(object))
    assert_to_from_json(np.dtype("b"))


def test_numpy_random_state():
    s1 = np.random.RandomState(seed=0)
    s2 = to_from_json(s1)
    assert s1.rand() == s2.rand()
    assert s1.rand() == s2.rand()  # twice on purpose
