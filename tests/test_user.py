# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

"""User-provided custom (de)serialization functions test suite"""


from datetime import datetime
from json import dumps
from typing import Any

import pytest
from common import assert_to_from_json, to_from_json

from turbo_broccoli import (
    Context,
    from_json,
    register_decoder,
    register_encoder,
    to_json,
)


class C:
    """Class with standard types as attributes"""

    a: int
    b: int

    def __init__(self, a: int, b: int):
        self.a, self.b = a, b

    def __eq__(self, other: object) -> bool:
        return isinstance(other, C) and self.a == other.a and self.b == other.b


class D:
    """
    Same as `C` but different name, so that instances of `C` are not instances
    of `D`.
    """

    a: int
    b: int

    def __init__(self, a: int, b: int):
        self.a, self.b = a, b

    def __eq__(self, other: object) -> bool:
        return isinstance(other, D) and self.a == other.a and self.b == other.b


class E:
    """Class with a nested arbitrary type"""

    a: int
    b: int
    c: Any

    def __init__(self, a: int, b: int, c: Any):
        self.a, self.b, self.c = a, b, c

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, E)
            and self.a == other.a
            and self.b == other.b
            and self.c.__class__.__name__ == other.c.__class__.__name__
            and self.c == other.c
        )


def encoder_c(obj: C, ctx: Context) -> dict:
    return {"__type__": "user.C", "a": obj.a, "b": obj.b}


def encoder_e(obj: C, ctx: Context) -> dict:
    return {"__type__": "user.E", "a": obj.a, "b": obj.b, "c": obj.c}


def encoder_e_skip(obj: C, ctx: Context) -> dict:
    return {
        "__type__": "user.E",
        "a": obj.a,
        "b": obj.b,
        "c": (obj.c.a, obj.c.b),
    }


def decoder_c(obj: dict, ctx: Context) -> C:
    return C(obj["a"], obj["b"])


def decoder_e(obj: dict, ctx: Context) -> E:
    return E(obj["a"], obj["b"], obj["c"])


def decoder_e_skip(obj: dict, ctx: Context) -> E:
    return E(obj["a"], obj["b"], C(obj["c"][0], obj["c"][1]))


def test_user_non_registered_1():
    with pytest.raises(TypeError):
        to_json(C(1, 2), Context())


def test_user_non_registered_2():
    register_encoder(encoder_c, C)
    register_decoder(decoder_c, C)
    with pytest.raises(TypeError):
        to_json(D(3, 4), Context())


def test_user():
    register_encoder(encoder_c, C)
    register_decoder(decoder_c, C)
    assert_to_from_json(C(1, 2))


def test_user_register_name():
    register_encoder(encoder_c, "C")
    register_decoder(decoder_c, "C")
    assert_to_from_json(C(1, 2))


def test_user_register_multiple():
    register_encoder(encoder_c, (C, D))
    register_decoder(decoder_c, (C, D))
    assert_to_from_json(C(1, 2))
    y = to_from_json(D(3, 4))
    assert y == C(3, 4)


def test_user_only_decoder():
    register_decoder(decoder_c, C)
    c = C(1, 2)
    d = dumps({"__type__": "user.C", "a": c.a, "b": c.b})
    assert c == from_json(d, Context())


def test_user_nested_1():
    register_encoder(encoder_e, E)
    register_decoder(decoder_e, E)
    assert_to_from_json(E(1, 2, datetime.now()))


def test_user_nested_2():
    register_encoder(encoder_c, C)
    register_decoder(decoder_c, C)
    register_encoder(encoder_e, E)
    register_decoder(decoder_e, E)
    assert_to_from_json(E(1, 2, C(3, 4)))


def test_user_nested_3():
    register_encoder(encoder_e_skip, E)
    register_decoder(decoder_e_skip, E)
    assert_to_from_json(E(1, 2, C(3, 4)))
