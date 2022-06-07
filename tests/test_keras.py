# pylint: disable=missing-function-docstring
"""Keras (de)serialization test suite"""

from tensorflow import keras
import numpy as np
from numpy.testing import assert_array_equal

from common import from_json, to_json


def test_keras_model():
    x = keras.Sequential(
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
    x.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    y = from_json(to_json(x))
    assert x.get_config() == y.get_config()
    for i, w in enumerate(x.weights):
        assert_array_equal(w, y.weights[i])
    # Not really necessary but why not
    assert_array_equal(x(np.ones((1, 28, 28, 1))), y(np.ones((1, 28, 28, 1))))
