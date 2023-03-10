"""Guarded call"""

from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

import json
from typing import Any, Callable, Dict, Union

from .turbo_broccoli import TurboBroccoliDecoder, TurboBroccoliEncoder


def produces_document(
    function: Callable[..., Dict[str, Any]], path: Union[str, Path]
) -> Callable[..., Dict[str, Any]]:
    """
    Consider an expensive function `f` that returns a TurboBroccoli/JSON-izable
    `dict`. Wrapping/decorating `f` using `produces_document` essentially saves
    the result at a specified path and when possible, loads it instead of
    calling `f`. For example:

    ```py
    _f = produces_document(f, "out/result.json")
    _f(*args, **kwargs)
    ```

    will only call `f` if the `out/result.json` does not exist, and otherwise,
    loads and returns `out/result.json`. However, if `out/result.json` exists
    and was produced by calling `_f(*args, **kwargs)`, then

    ```py
    _f(*args2, **kwargs2)
    ```

    will still return the same result. Therefore, this wrapper isn't
    recommended if you intend to call `f` many times with different arguments.
    """

    def _wrapped(*args, **kwargs) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fp:
                obj = json.load(fp, cls=TurboBroccoliDecoder)
            logging.debug(
                f"Skipped call to guarded method '{function.__name__}'"
            )
            return obj
        except:  # pylint: disable=bare-except
            obj = function(*args, **kwargs)
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(obj, fp, cls=TurboBroccoliEncoder)
            return obj

    return _wrapped
