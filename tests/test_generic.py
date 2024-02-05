# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

from common import to_from_json


# pylint: disable=missing-class-docstring
class C:
    __turbo_broccoli__ = ["a_str", "an_int"]
    a_byte_str: bytes
    a_list: list
    a_str: str

    @property
    def an_int(self):
        return 2 * len(self.a_list)


def test_dataclass():
    x = C()
    x.a_byte_str = "🦐🦐🦐".encode("utf8")
    x.a_list = list(range(10))
    x.a_str = "Hello 🌎"
    y = {"a_str": x.a_str, "an_int": x.an_int}
    z = to_from_json(x)
    assert y == z
