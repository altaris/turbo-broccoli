# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

from common import from_json, to_json


def test_bytes_empty():
    x = b""
    assert x == from_json(to_json(x))


def test_bytes_ascii():
    x = "Hello".encode("ascii")
    assert x == from_json(to_json(x))


def test_bytes_utf8():
    x = "Hello 👋".encode("utf8")
    assert x == from_json(to_json(x))
