# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from json import loads

from turbo_broccoli import EmbeddedDict, EmbeddedList, from_json, to_json


def test_embedded_dict():
    x = {"a": 1, "b": EmbeddedDict({"c": 2, "d": 3})}
    u = to_json(x)
    v, y = loads(u), from_json(u)
    assert isinstance(y["b"], EmbeddedDict)
    assert set(v.keys()) == {"a", "b"}
    assert v["a"] == 1
    assert set(v["b"].keys()) == {"__type__", "__version__", "id"}
    assert v["b"]["__type__"] == "embedded.dict"
    assert x == y


def test_embedded_list():
    x = [1, EmbeddedList([2, 3])]
    u = to_json(x)
    v, y = loads(u), from_json(u)
    assert isinstance(y[1], EmbeddedList)
    assert v[0] == 1
    assert isinstance(v[1], dict)
    assert set(v[1].keys()) == {"__type__", "__version__", "id"}
    assert v[1]["__type__"] == "embedded.list"
    assert x == y


def test_embedded_dict_double_save():
    x = {"a": 1, "b": EmbeddedDict({"c": 2, "d": 3})}
    u1 = to_json(x)
    v1, y1 = loads(u1), from_json(u1)
    id1 = v1["b"]["id"]
    u2 = to_json(y1)
    v2, y2 = loads(u2), from_json(u2)
    id2 = v2["b"]["id"]
    assert id1 == id2
    assert id1 == y1["b"]._tb_artifact_id
    assert id2 == y2["b"]._tb_artifact_id
    assert x == y1
    assert x == y2


def test_embedded_list_double_save():
    x = [1, EmbeddedList([2, 3])]
    u1 = to_json(x)
    v1, y1 = loads(u1), from_json(u1)
    id1 = v1[1]["id"]
    u2 = to_json(y1)
    v2, y2 = loads(u2), from_json(u2)
    id2 = v2[1]["id"]
    assert id1 == id2
    assert id1 == y1[1]._tb_artifact_id
    assert id2 == y2[1]._tb_artifact_id
    assert x == y1
    assert x == y2
