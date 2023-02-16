# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from dataclasses import dataclass

from turbo_broccoli.environment import register_dataclass_type

from common import from_json, to_json


@dataclass
class C:
    a_byte_str: bytes
    a_list: list
    a_str: str
    an_int: int


@dataclass
class D:
    a_dataclass: C
    a_float: float


def test_dataclass():
    register_dataclass_type(C)
    x = C(
        a_byte_str="🦐🦐🦐".encode("utf8"),
        a_list=list(range(10)),
        a_str="Hello 🌎",
        an_int="42",
    )
    y = from_json(to_json(x))
    assert isinstance(y, C)
    assert x.__dict__ == y.__dict__


def test_dataclass_recursive():
    register_dataclass_type(C)
    register_dataclass_type(D)
    x = D(
        a_dataclass=C(
            a_byte_str="🦐🦐🦐".encode("utf8"),
            a_list=list(range(10)),
            a_str="Hello 🌎",
            an_int="42",
        ),
        a_float=1.2,
    )
    y = from_json(to_json(x))
    assert isinstance(y, D)
    assert x.a_float == y.a_float
    assert isinstance(x.a_dataclass, C)
    assert x.a_dataclass.__dict__ == y.a_dataclass.__dict__
