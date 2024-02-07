# pylint: disable=missing-function-docstring
"""(de)serialization of secrets"""

import nacl.secret
import nacl.utils
import pytest
from common import to_from_json
from nacl.exceptions import CryptoError

from turbo_broccoli import (
    Context,
    LockedSecret,
    Secret,
    SecretDict,
    SecretFloat,
    SecretInt,
    SecretList,
    SecretStr,
    from_json,
    to_json,
)


def _new_key() -> bytes:
    return nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)


def assert_secret_equal(a: Secret, b: Secret):
    assert type(a) == type(b)  # pylint: disable=unidiomatic-typecheck
    if not isinstance(a, LockedSecret):
        assert a.get_secret_value() == b.get_secret_value()


def test_secret():
    ctx = Context(nacl_shared_key=_new_key())
    x = {
        "a_dict": SecretDict({"a": 1, "b": 2}),
        "a_float": SecretFloat(1.2),
        "a_int": SecretInt(12),
        "a_list": SecretList(list(range(10))),
        "a_str": SecretStr("password"),
    }
    y = to_from_json(x, ctx)
    assert_secret_equal(x["a_dict"], y["a_dict"])
    assert_secret_equal(x["a_float"], y["a_float"])
    assert_secret_equal(x["a_int"], y["a_int"])
    assert_secret_equal(x["a_list"], y["a_list"])
    assert_secret_equal(x["a_str"], y["a_str"])


def test_secret_nokey():
    ctx = Context(nacl_shared_key=_new_key())
    x = {
        "a_dict": SecretDict({"a": 1, "b": 2}),
        "a_float": SecretFloat(1.2),
        "a_int": SecretInt(12),
        "a_list": SecretList(list(range(10))),
        "a_str": SecretStr("password"),
    }
    y = to_json(x, ctx)
    z = from_json(y)
    assert isinstance(z["a_dict"], LockedSecret)
    assert isinstance(z["a_float"], LockedSecret)
    assert isinstance(z["a_int"], LockedSecret)
    assert isinstance(z["a_list"], LockedSecret)
    assert isinstance(z["a_str"], LockedSecret)


def test_secret_wrongkey():
    ctx = Context(nacl_shared_key=_new_key())
    x = {
        "a_dict": SecretDict({"a": 1, "b": 2}),
        "a_float": SecretFloat(1.2),
        "a_int": SecretInt(12),
        "a_list": SecretList(list(range(10))),
        "a_str": SecretStr("password"),
    }
    y = to_json(x, ctx)
    ctx = Context(nacl_shared_key=_new_key())
    with pytest.raises(CryptoError):
        from_json(y, ctx)
