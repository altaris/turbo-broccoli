# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

from common import assert_to_from_json


def test_bytes_empty():
    assert_to_from_json(b"")


def test_bytes_ascii():
    assert_to_from_json("Hello".encode("ascii"))


def test_bytes_utf8():
    assert_to_from_json("Hello 👋".encode("utf8"))
