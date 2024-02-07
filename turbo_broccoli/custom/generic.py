"""
Serialization of so-called generic object. See
`turbo_broccoli.generic.to_json`.
"""

from typing import Any, Iterable

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import TypeNotSupported


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a generic object into JSON. The return document contains all
    attributes listed in the object's `__turbo_broccoli__` attribute.
    """
    if not (
        hasattr(obj, "__turbo_broccoli__")
        and isinstance(obj.__turbo_broccoli__, Iterable)
    ):
        raise TypeNotSupported()
    return {k: getattr(obj, k) for k in obj.__turbo_broccoli__}
