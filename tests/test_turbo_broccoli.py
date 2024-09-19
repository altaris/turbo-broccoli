"""bytes (de)serialization test suite"""

import json

import pytest

from turbo_broccoli import Context, load_json, save_json

TEST_PATH = "out/test/test_turbo_broccoli/"


def test_turbo_broccoli_save_json_file_path():
    p = TEST_PATH + "test_turbo_broccoli_save_json_file_path.json"
    x = {"a": 1, "b": 2}
    save_json(x, p)


def test_turbo_broccoli_save_json_context():
    p = TEST_PATH + "test_turbo_broccoli_save_json_context.json"
    x = {"a": 1, "b": 2}
    save_json(x, ctx=Context(p))


def test_turbo_broccoli_save_json_none():
    x = {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        save_json(x)


def test_turbo_broccoli_save_json_conflict():
    x = {"a": 1, "b": 2}
    with pytest.raises(ValueError):
        save_json(x, "a.json", Context("b.json"))


def test_turbo_broccoli_load_json_file_path():
    p = TEST_PATH + "test_turbo_broccoli_load_json_file_path.json"
    x = {"a": 1, "b": 2}
    with open(p, mode="w", encoding="utf-8") as fp:
        json.dump(x, fp)
    load_json(p)


def test_turbo_broccoli_load_json_context():
    p = TEST_PATH + "test_turbo_broccoli_load_json_context.json"
    x = {"a": 1, "b": 2}
    with open(p, mode="w", encoding="utf-8") as fp:
        json.dump(x, fp)
    load_json(ctx=Context(p))


def test_turbo_broccoli_load_json_none():
    with pytest.raises(ValueError):
        load_json()


def test_turbo_broccoli_load_json_conflict():
    with pytest.raises(ValueError):
        load_json("a.json", Context("b.json"))


def test_turbo_broccoli_save_load_json_compress():
    p = TEST_PATH + "test_turbo_broccoli_save_load_json_compress.json.gz"
    x = {"a": "abc" * 1000}
    save_json(x, p)
    y = load_json(p)
    assert x == y
