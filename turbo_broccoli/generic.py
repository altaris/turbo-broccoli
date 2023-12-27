"""
Serialization of so-called generic object. See
`turbo_broccoli.generic.to_json`.
"""


from typing import Any, Iterable

from turbo_broccoli.utils import TypeNotSupported, raise_if_nodecode


def to_json(obj: Any) -> dict:
    """
    Serializes a generic object into JSON. The return document contains all
    attributes listed in the object's `__turbo_broccoli__` attribute.

    """
    if not (
        hasattr(obj, "__turbo_broccoli__")
        and isinstance(obj.__turbo_broccoli__, Iterable)
    ):
        raise TypeNotSupported()
    raise_if_nodecode("generic")
    return {k: getattr(obj, k) for k in obj.__turbo_broccoli__}
