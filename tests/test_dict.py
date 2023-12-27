# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

import json

from common import assert_to_from_json, from_json, to_json


def test_str_dict():
    x = {"a": 1, "b": 2}
    assert_to_from_json(x)
    assert x == json.loads(json.dumps(x))


def test_int_dict():
    x, y = {1: "a", 2: "b"}, {"1": "a", "2": "b"}
    assert_to_from_json(x)
    assert y == json.loads(json.dumps(x))


def test_float_dict():
    x = {1.1: "a", 2.2: "b"}
    assert_to_from_json(x)


def test_composite_dict():
    x = {1: "a", "2": "b", 1e5: "c"}
    assert_to_from_json(x)
