"""Various utilities and internal methods"""

import re
from typing import Any, Generator


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


def artifacts(doc: Any) -> Generator[str, None, None]:
    """
    Lists all the artifacts names referenced by this JSON document. Obviously,
    it should have been deserialized using vanilla `json.load` or `json.loads`,
    or using turbo broccoli with adequate nodecodes.

    In reality, this method recursively traverses the document and searches for
    dicts that:

    - have a `"__version__"`, `"__type__"`, and a `"id"` field;

    - the value at `"id"` is a UUID4 or has the form `<uuid4>.<...>`, i.e.
      matches the regexp

        ```re
        ^[0-9a-f]{8}(\\-[0-9a-f]{4}){3}\\-[0-9a-f]{12}(\\..+)?$
        ```

      in which case that value is `yield`ed.

    TODO:
        Implement a smarter way
    """
    if isinstance(doc, dict):
        fields = ["__version__", "__type__", "id"]
        if all(map(lambda f: f in doc, fields)):
            v = doc["id"]
            r = re.compile(
                r"^[0-9a-f]{8}(\-[0-9a-f]{4}){3}\-[0-9a-f]{12}(\..+)?$"
            )
            if r.match(v):
                yield v + ".tb"
        else:
            for v in doc.values():
                for a in artifacts(v):
                    yield a
    elif isinstance(doc, list):
        for x in doc:
            for a in artifacts(x):
                yield a
