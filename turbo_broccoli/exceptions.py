"""Various utilities and internal methods"""


class DeserializationError(Exception):
    """Raised whenever something went wrong during deserialization"""


class SerializationError(Exception):
    """Raised whenever something went wrong during serialization"""


class TypeNotSupported(Exception):
    """
    `to_json` will raise that if they are fed types they cannot manage. This is
    fine, the dispatch in `turbo_broccoli.turbo_broccoli._to_jsonable` catches
    these and moves on to the next registered `to_json` method.
    """


class TypeIsNodecode(Exception):
    """Raised during deserialization if the type shouldn't be decoded"""
