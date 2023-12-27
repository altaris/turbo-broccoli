"""Common testing utilities"""

import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))


from turbo_broccoli import TurboBroccoliDecoder, TurboBroccoliEncoder


def assert_to_from_json(obj: Any):
    """
    Shorthand for
    ```py
    assert obj == from_json(to_json(obj))
    ```
    """
    assert obj == from_json(to_json(obj))


def from_json(doc: str) -> Any:
    """Converts a JSON document back to a Python object"""
    return json.loads(doc, cls=TurboBroccoliDecoder)


def to_json(obj: Any) -> str:
    """Converts an object to JSON"""
    return json.dumps(obj, cls=TurboBroccoliEncoder)
