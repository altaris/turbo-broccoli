"""Guarded call"""

import hashlib
from functools import partial
from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from typing import Any, Callable, Dict, Optional, Union

from .turbo_broccoli import (
    load_json,
    save_json,
    to_json,
)


class GuardedBlockHandler:
    """
    A guarded block handler allows to guard an entire block of code. Use it as
    follows:

        ```py
        h = GuardedBlockHandler("foo.json")
        for _ in h.guard():
            # This whole block will be skipped if foo.json exists
            # If not, don't forget to set the results:
            h.result = ...
        # In any case, the results of the block are available in h.result
        ```

    The handler's `result` is `None` by default. If left to `None`, no output
    file is created. This allows for scenarios like

        ```py
        h = GuardedBlockHandler("foo.json")
        for _ in h.guard():
            ... # Guarded code
            if success:
                h.result = ...
        ```

    So if the guarded code did not succeed, then `foo.json` is not created,
    and so the next time, it will be run again.
    """

    name: Optional[str]
    result: Any = None
    output_file: Path

    def __init__(
        self, output_path: Union[str, Path], name: Optional[str] = None
    ) -> None:
        self.output_file = Path(output_path)
        self.name = name

    def _load(self):
        """Loads the results and logs"""
        self.result = load_json(self.output_file)
        if self.name:
            logging.debug(f"Skipped guarded block '{self.name}'")

    def _save(self):
        """Saves the results (if not `None`) and logs"""
        if self.result is None:
            return
        save_json(self.result, self.output_file)
        if self.name is not None:
            logging.debug(
                f"Saved guarded block '{self.name}'s results to "
                f"'{self.output_file}'"
            )

    def guard(self):
        """See `turbo_broccoli.guard.GuardedBlockHandler`'s documentation"""
        if self.output_file.is_file():
            self._load()
        else:
            try:
                yield
            finally:
                self._save()


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
            obj = load_json(path)
            logging.debug(
                f"Skipped call to guarded method '{function.__name__}'"
            )
            return obj
        except:  # pylint: disable=bare-except
            obj = function(*args, **kwargs)
            save_json(obj, path)
            return obj

    path = path if isinstance(path, Path) else Path(path)
    return partial(_wrapped, path)
