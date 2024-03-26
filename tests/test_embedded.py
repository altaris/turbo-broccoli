# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from json import loads

from turbo_broccoli import EmbeddedDict, EmbeddedList, from_json, to_json


def test_embedded_dict():
    x = {"a": 1, "b": EmbeddedDict({"c": 2, "d": 3})}
    u = to_json(x)
    v, y = loads(u), from_json(u)
    assert set(v.keys()) == {"a", "b"}
    assert v["a"] == 1
    assert set(v["b"].keys()) == {"__type__", "__version__", "id"}
    assert v["b"]["__type__"] == "embedded.dict"
    assert x == y


def test_embedded_list():
    x = [1, EmbeddedList([2, 3])]
    u = to_json(x)
    v, y = loads(u), from_json(u)
    assert v[0] == 1
    assert isinstance(v[1], dict)
    assert set(v[1].keys()) == {"__type__", "__version__", "id"}
    assert v[1]["__type__"] == "embedded.list"
    assert x == y
