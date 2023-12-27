"""
Serialization of so-called generic object. See
`turbo_broccoli.generic.to_json`.
"""
__docformat__ = "google"


from typing import Any, Iterable

from turbo_broccoli.utils import TypeNotSupported, raise_if_nodecode


def to_json(obj: Any) -> dict:
    """
    Serializes a generic object into JSON by cases. The return dict has the
    following structure:

        {
            "__type__": "generic",
            "__version__": 2,
            "data": {...},
        }

    """
    if not (
        hasattr(obj, "__turbo_broccoli__")
        and isinstance(obj.__turbo_broccoli__, Iterable)
    ):
        raise TypeNotSupported()
    raise_if_nodecode("generic")
    return {
        "__type__": "generic",
        "__version__": 2,
        "data": {k: getattr(obj, k) for k in obj.__turbo_broccoli__},
    }
