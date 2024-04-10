# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

from pathlib import Path

from common import assert_to_from_json, to_from_json


def test_pathlib_path_1():
    p = Path("a/b/c")
    q = to_from_json(p)
    assert p == q
    assert str(p) == str(q)


def test_pathlib_path_2():
    assert_to_from_json(Path("/"))


def test_pathlib_path_3():
    assert_to_from_json(Path("."))


def test_pathlib_path_4():
    assert_to_from_json(Path("../.."))


def test_pathlib_path_5():
    assert_to_from_json(Path("/.."))
