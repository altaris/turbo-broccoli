# pylint: disable=missing-function-docstring
"""Keras (de)serialization test suite"""

import numpy as np
from common import to_from_json
from numpy.testing import assert_array_equal
from tensorflow import keras  # pylint: disable=no-name-in-module

from turbo_broccoli import Context


def _assert_model_equal(a, b):
    for i, w in enumerate(a.weights):
        assert_array_equal(w, b.weights[i])
    # Not really necessary but why not
    assert_array_equal(a(np.ones((1, 28, 28, 1))), b(np.ones((1, 28, 28, 1))))


def _build_model():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def test_keras_model_json():
    ctx = Context(keras_format="json")
    m1 = _build_model()
    m2 = to_from_json(m1, ctx)
    _assert_model_equal(m1, m2)
    assert m1.get_config() == m2.get_config()


def test_keras_model_h5():
    ctx = Context(keras_format="h5")
    m = _build_model()
    _assert_model_equal(m, to_from_json(m, ctx))


def test_keras_model_keras():
    ctx = Context(keras_format="keras")
    m = _build_model()
    _assert_model_equal(m, to_from_json(m, ctx))


def test_keras_model_tf():
    ctx = Context(keras_format="tf")
    m = _build_model()
    _assert_model_equal(m, to_from_json(m, ctx))
