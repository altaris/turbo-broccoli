# pylint: disable=missing-class-docstring
"""Serialize secrets"""
__docformat__ = "google"

import json
from typing import Any, NoReturn

from nacl.secret import SecretBox

from turbo_broccoli.environment import get_shared_key


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


def _from_json_v1(dct: dict) -> Any:
    """
    Converts a JSON document to a Python type following the v1 specification.
    """
    key = get_shared_key()
    if key is None:
        return LockedSecret()
    box = SecretBox(key)
    return json.loads(box.decrypt(dct["data"]).decode("utf-8"))


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a secret Python type. See `to_json` for the
    specification `dct` is expected to follow. In particular, note that `dct`
    must contain the key `__secret__`.
    """
    DECODERS = {
        1: _from_json_v1,
    }
    obj = DECODERS[dct["__secret__"]["__version__"]](dct["__secret__"])
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


def to_json(obj: Secret) -> dict:
    """
    Encrypts a JSON **string representation** of a secret document into a new
    JSON document with the following structure:

        {
            "__secret__": {
                "__version__": 1,
                "data": <encrypted bytes>,
            }
        }
    """
    if not isinstance(obj, Secret):
        raise TypeError("Not a valid secret type")
    key = get_shared_key()
    if key is None:
        raise RuntimeError(
            "Attempting to serialize a secret type but no shared key is set. "
            "Use either turbo_broccoli.environment.set_shared_key or the "
            "TB_SHARED_KEY environment variable."
        )
    box = SecretBox(key)
    return {
        "__secret__": {
            "__version__": 1,
            "data": box.encrypt(
                json.dumps(obj.get_secret_value()).encode("utf-8")
            ),
        }
    }
