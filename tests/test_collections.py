# pylint: disable=missing-function-docstring
"""Python collections (de)serialization test suite"""

from collections import deque, namedtuple
from typing import NamedTuple

from common import assert_to_from_json, to_from_json


def _assert_equal(a: deque, b: deque):
    assert a.maxlen == b.maxlen
    assert a == b


def test_deque_empty():
    x = deque()
    _assert_equal(x, to_from_json(x))
    x = deque([])
    _assert_equal(x, to_from_json(x))


def test_deque_normal():
    x = deque(range(100))
    _assert_equal(x, to_from_json(x))


def test_deque_maxlen_empty():
    x = deque([], maxlen=1)
    _assert_equal(x, to_from_json(x))


def test_deque_maxlen_trunc():
    x = deque(range(100), maxlen=10)
    _assert_equal(x, to_from_json(x))


def test_namedtuple_empty():
    C = namedtuple("C", [])
    assert_to_from_json(C())


def test_namedtuple_normal():
    C = namedtuple("C", ["x", "y"])
    assert_to_from_json(C(1, 2))


def test_namedtuple_default():
    C = namedtuple("C", ["x", "y"], defaults=[0, 1])
    assert_to_from_json(C())


def test_namedtuple_subclass_1():
    # pylint: disable=missing-class-docstring
    class C(namedtuple("C", ["x", "y"])):
        @property
        def z(self):
            return self.x + self.y

    assert_to_from_json(C(1, 2))


def test_namedtuple_subclass_2():
    # pylint: disable=missing-class-docstring
    class C(NamedTuple):
        x: int
        y: int

        @property
        def z(self):
            return self.x + self.y

    assert_to_from_json(C(1, 2))


def test_set():
    assert_to_from_json({1, 2, 3})


def test_set_empty():
    assert_to_from_json({})


def test_tuple():
    assert_to_from_json((1, 2, 3))
