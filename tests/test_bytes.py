# pylint: disable=missing-function-docstring
"""bytes (de)serialization test suite"""

from common import assert_to_from_json

from turbo_broccoli import Context, from_json, to_json


def test_bytes_empty():
    assert_to_from_json(b"")


def test_bytes_ascii():
    assert_to_from_json("Hello".encode("ascii"))


def test_bytes_utf8():
    assert_to_from_json("Hello 👋".encode("utf8"))


def test_bytes_large():
    x = ("👋" * 1000).encode("utf8")
    ctx = Context(min_artifact_size=0)
    doc = to_json(x, ctx)
    assert "id" in doc
    assert "data" not in doc
    assert x == from_json(doc, ctx)
