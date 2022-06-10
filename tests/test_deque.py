# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from collections import deque

from common import from_json, to_json


def _assert_equal(a: deque, b: deque):
    assert a.maxlen == b.maxlen
    assert a == b


def test_deque_empty():
    x = deque()
    _assert_equal(x, from_json(to_json(x)))
    x = deque([])
    _assert_equal(x, from_json(to_json(x)))


def test_deque_normal():
    x = deque(range(100))
    _assert_equal(x, from_json(to_json(x)))


def test_deque_maxlen_empty():
    x = deque([], maxlen=1)
    _assert_equal(x, from_json(to_json(x)))


def test_deque_maxlen_trunc():
    x = deque(range(100), maxlen=10)
    _assert_equal(x, from_json(to_json(x)))
