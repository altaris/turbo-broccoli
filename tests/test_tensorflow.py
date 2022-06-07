# pylint: disable=missing-function-docstring
"""Tensorflow (de)serialization test suite"""

import tensorflow as tf

from common import from_json, to_json


def _assert_equal(a, b):
    assert type(a) == type(b)  # pylint: disable=unidiomatic-typecheck
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    tf.debugging.assert_equal(a, b)


def test_tensorflow_numerical():
    x = tf.constant([])
    _assert_equal(x, from_json(to_json(x)))
    x = tf.constant([1, 2, 3])
    _assert_equal(x, from_json(to_json(x)))
    x = tf.random.uniform((10, 10))
    _assert_equal(x, from_json(to_json(x)))

