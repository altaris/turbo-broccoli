# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from dataclasses import dataclass

from common import from_json, to_json

# pylint: disable=missing-class-docstring
@dataclass
class C:
    a_byte_str: bytes
    a_list: list
    a_str: str
    an_int: int


def test_dataclass():
    x = C(
        a_byte_str="🦐🦐🦐".encode("utf8"),
        a_list=list(range(10)),
        a_str="Hello 🌎",
        an_int="42",
    )
    y = from_json(to_json(x))["__dataclass__"]["data"]
    assert x.__dict__ == y
