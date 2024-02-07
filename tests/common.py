"""Common testing utilities"""

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent))


# pylint: disable=wrong-import-position
from turbo_broccoli import Context, from_json, to_json


def assert_to_from_json(obj: Any, ctx: Context | None = None):
    """
    Shorthand for
    ```py
    assert obj == from_json(to_json(obj))
    ```
    """
    assert obj == from_json(to_json(obj, ctx), ctx)


def to_from_json(obj: Any, ctx: Context | None = None):
    """
    Shorthand for
    ```py
    assert obj == from_json(to_json(obj))
    ```
    """
    return from_json(to_json(obj, ctx), ctx)
