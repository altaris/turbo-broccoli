# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

from json import loads

from common import assert_to_from_json

from turbo_broccoli import from_json, to_json


def test_dict_normal():
    x = {"a": 1, "b": 2}
    u = to_json(x)
    v, y = loads(u), from_json(u)
    assert x == y
    assert x == v


def test_dict():
    # pylint: disable=duplicate-key
    x = {
        "a": "a",
        1: 1,
        2.3: 2.3,
        True: True,
        False: False,
        None: None,
        (1, 2): (1, 2),
        b"abc": b"abc",
    }
    assert_to_from_json(x)
