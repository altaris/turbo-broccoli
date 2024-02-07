# pylint: disable=missing-function-docstring
"""Tensorflow (de)serialization test suite"""

import tensorflow as tf
from common import to_from_json

from turbo_broccoli import Context


def _assert_sparse_equal(a, b):
    tf.debugging.assert_equal(a.dense_shape, b.dense_shape)
    tf.debugging.assert_equal(a.indices, b.indices)
    tf.debugging.assert_equal(a.values, b.values)


def test_tensorflow_numerical():
    x = tf.constant([])
    tf.debugging.assert_equal(x, to_from_json(x))
    x = tf.constant([1, 2, 3])
    tf.debugging.assert_equal(x, to_from_json(x))
    x = tf.random.uniform((10, 10))
    tf.debugging.assert_equal(x, to_from_json(x))


def test_tensorflow_numerical_large():
    ctx = Context(min_artifact_size=0)
    x = tf.random.uniform((100, 100), dtype=tf.float64)
    tf.debugging.assert_equal(x, to_from_json(x, ctx))


def test_tensorflow_sparse():
    x = tf.SparseTensor(indices=[[0, 0]], values=[1], dense_shape=(1, 1))
    _assert_sparse_equal(x, to_from_json(x))
    x = tf.SparseTensor(
        indices=[[0, 3], [2, 4]],
        values=[10, 20],
        dense_shape=(10_000_000_000, 10_000_000_000),
    )
    _assert_sparse_equal(x, to_from_json(x))


def test_tensorflow_variable():
    x = tf.Variable(1.0)
    tf.debugging.assert_equal(x, to_from_json(x))
    x = tf.Variable([1.0])
    tf.debugging.assert_equal(x, to_from_json(x))
    x = tf.Variable(tf.random.uniform((10, 10)))
    tf.debugging.assert_equal(x, to_from_json(x))
