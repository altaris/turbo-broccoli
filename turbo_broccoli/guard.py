"""Guarded call"""

from pathlib import Path
import hashlib
from functools import partial

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

import json
from typing import Any, Callable, Dict, Union

from .turbo_broccoli import TurboBroccoliDecoder, TurboBroccoliEncoder, to_json


def guarded_call(
    function: Callable[..., Dict[str, Any]],
    path: Union[str, Path],
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function:

    ```py
    guarded_call(f, "out/result.json", *args, **kwargs)
    ```
    is equivalent to

    ```py
    _f = produces_document(f, "out/result.json")
    _f(*args, **kwargs)
    ```
    """
    _f = produces_document(function, path)
    return _f(*args, **kwargs)


def produces_document(
    function: Callable[..., Dict[str, Any]],
    path: Union[str, Path],
    check_args: bool = False,
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

    will still return the same result. If you want to keep a different file for
    each `args`/`kwargs`, set `check_args` to `True` as in

    ```py
    _f = produces_document(f, "out/result.json")
    _f(*args, **kwargs)
    ```

    In this case, the arguments must be TurboBroccoli/JSON-izable, i.e. the
    document

    ```
    {
        "args": args,
        "kwargs": kwargs,
    }
    ```

    must be TurboBroccoli/JSON-izable. The resulting file is no longer
    `out/result.json` but rather `out/result.json/<hash>` where `hash` is the
    MD5 hash of the serialization of the `args`/`kwargs` document above.
    """

    def _wrapped(path: Path, *args, **kwargs) -> Dict[str, Any]:
        if check_args:
            s = to_json({"args": args, "kwargs": kwargs})
            h = hashlib.md5(s.encode("utf-8")).hexdigest()
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{h}.json"
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

    path = path if isinstance(path, Path) else Path(path)
    return partial(_wrapped, path)
