# pylint: disable=missing-class-docstring
"""Serialize secrets"""

import json
from typing import Any, NoReturn

from nacl.secret import SecretBox

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import TypeNotSupported


class Secret:
    """
    A wrapper for a basic Python variable whose value is considered to be
    secret. Similar API as [`pydantic`'s secret
    types](https://pydantic-docs.helpmanual.io/usage/types/#secret-types)
    """

    _value: Any

    def __eq__(self, __o: object) -> bool:
        return False

    def __init__(self, value: Any) -> None:
        self._value = value

    def __ne__(self, __o: object) -> bool:
        return False

    def __repr__(self) -> str:
        return "--REDACTED--"

    def __str__(self) -> str:
        return "--REDACTED--"

    def get_secret_value(self) -> Any:
        """Self-explanatory"""
        return self._value


class LockedSecret(Secret):
    """
    Represented a secret that could not be decrypted because the shared key was
    not provided. The `get_secret_value` method always raises a `RuntimeError`.
    """

    def __init__(self) -> None:
        super().__init__(None)

    def get_secret_value(self) -> NoReturn:
        raise RuntimeError("Cannot get the secret value of a locked secret")


class SecretDict(Secret):
    def __init__(self, value: dict) -> None:
        super().__init__(value)


class SecretFloat(Secret):
    def __init__(self, value: float) -> None:
        super().__init__(value)


class SecretInt(Secret):
    def __init__(self, value: int) -> None:
        super().__init__(value)


class SecretList(Secret):
    def __init__(self, value: list) -> None:
        super().__init__(value)


class SecretStr(Secret):
    def __init__(self, value: str) -> None:
        super().__init__(value)


def _from_json_v2(dct: dict, ctx: Context) -> Any:
    if ctx.nacl_shared_key is None:
        return LockedSecret()
    box = SecretBox(ctx.nacl_shared_key)
    return json.loads(box.decrypt(dct["data"]).decode("utf-8"))


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    ctx.raise_if_nodecode("bytes")
    DECODERS = {
        # 1: _from_json_v1,  # Use turbo_broccoli v3
        2: _from_json_v2,
    }
    obj = DECODERS[dct["__version__"]](dct, ctx)
    if isinstance(obj, LockedSecret):
        return obj
    TYPES = {
        dict: SecretDict,
        float: SecretFloat,
        int: SecretInt,
        list: SecretList,
        str: SecretStr,
    }
    return TYPES[type(obj)](obj)


def to_json(obj: Secret, ctx: Context) -> dict:
    """
    Encrypts a JSON **string representation** of a secret document into a
    new JSON document with the following structure:

    ```py
    {
        "__type__": "secret",
        "__version__": 2,
        "data": <encrypted bytes>,
    }
    ```
    """
    if not isinstance(obj, Secret):
        raise TypeNotSupported()
    if ctx.nacl_shared_key is None:
        raise RuntimeError(
            "Attempting to serialize a secret type but no shared key is set. "
            "Either set `nacl_shared_key` when constructing the encoding "
            "torbo_broccoli.context.Context, or set the TB_SHARED_KEY "
            "environment variable."
        )
    box = SecretBox(ctx.nacl_shared_key)
    return {
        "__type__": "secret",
        "__version__": 2,
        "data": box.encrypt(
            json.dumps(obj.get_secret_value()).encode("utf-8")
        ),
    }
