"""Guarded call"""

import hashlib
from functools import partial
from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from typing import Any, Callable, Dict, Generator, Iterable, Optional, Union

from .turbo_broccoli import load_json, save_json, to_json


class GuardedBlockHandler:
    """
    A guarded block handler allows to guard an entire block/loop of code.

    ### Guarded block

    Use it as follows:

    ```py
    h = GuardedBlockHandler("out/foo.json")
    for _ in h.guard():
        # This whole block will be skipped if out/foo.json exists
        # If not, don't forget to set the results:
        h.result = ...
    # In any case, the results of the block are available in h.result
    ```

    The handler's `result` is `None` by default. If left to `None`, no output
    file is created. This allows for scenarios like

    ```py
    h = GuardedBlockHandler("out/foo.json")
    for _ in h.guard():
        ... # Guarded code
        if success:
            h.result = ...
    ```

    So if the guarded code did not succeed, then `out/foo.json` is not created,
    and so the next time, it will be run again.

    Note that the parent directory of the output file (in this case, `out/`)
    will be created if it does not exist.

    ### Guarded loop

    If an iterable `l` is passed to `h.guard`, then `for _ in h.guard(l):`
    iterates over `l`, and after every iteration, an artifact is created using
    the current value of `h.result` at `<output_path>/<str(element)>`. For
    example:

    ```py
    l = [1, 2, 3]  # The elements of l must be stringable
    h = GuardedBlockHandler("out/foo")
    for x in h.guard(l):
        # At the begining of every iteration, h.result is guaranteed to be a
        # dict with key str(x) being None. The result of past iterations are
        # also in h.result
        h.result[x] = {"bar": int(x) + 1}
    # now, h.result is {"1": {"bar": 2}, "2": {"bar": 3}, ...}
    ```

    creates folder `out/foo` in which there will be files `1.json`, `2.json`,
    `3.json` respectively containing `{"bar": 2}`, `{"bar": 3}`, `{"bar": 4}`.
    Of course, if any of these files existed prior to running the `for` loop
    (e.g. `out/foo/2.json`), then the corresponding iteration (in that case `x
    = 2`) is skipped.

    **Warning**: The `x` in the loop above are actually strings! More generally,
    `h.guard(l)` actually iterates over `map(str, l)`.
    """

    name: Optional[str]
    result: Any = None
    output_path: Path

    def __init__(
        self, output_path: Union[str, Path], name: Optional[str] = None
    ) -> None:
        self.output_path = Path(output_path)
        self.name = name

    def _guard_iter(self, iterable: Iterable[Any]) -> Generator:
        """
        Internal implementation of a guarded loop. See
        `turbo_broccoli.guard.GuardedBlockHandler`'s documentation.
        """
        self.result = {}
        for x in iterable:
            sx = str(x)
            path = self.output_path / f"{sx}.json"
            if path.is_file():
                self.result[sx] = load_json(path)
                if self.name:
                    logging.debug(
                        f"Skipped guarded iteration '{self.name}'[{sx}]"
                    )
            else:
                self.result[sx] = None
                yield sx
                if self.result[sx] is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    save_json(self.result[sx], path)
                    if self.name is not None:
                        logging.debug(
                            f"Saved guarded iteration '{self.name}'[{sx}] "
                            f"results to '{path}'"
                        )

    def _guard_no_iter(self) -> Generator:
        """
        Internal implementation of a guarded block. See
        `turbo_broccoli.guard.GuardedBlockHandler`'s documentation.
        """
        if self.output_path.is_file():
            self.result = load_json(self.output_path)
            if self.name:
                logging.debug(f"Skipped guarded block '{self.name}'")
        else:
            try:
                yield
            finally:
                if self.result is not None:
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                    save_json(self.result, self.output_path)
                    if self.name is not None:
                        logging.debug(
                            f"Saved guarded block '{self.name}' results to "
                            f"'{self.output_path}'"
                        )

    def guard(self, iterable: Optional[Iterable[Any]] = None) -> Generator:
        """See `turbo_broccoli.guard.GuardedBlockHandler`'s documentation"""
        if iterable is None:
            for _ in self._guard_no_iter():
                yield
        else:
            for x in self._guard_iter(iterable):
                yield x


def guarded_call(
    function: Callable[..., Any],
    path: Union[str, Path],
    *args,
    **kwargs,
) -> Any:
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

    Note that the parent directory of the output file (in this case, `out/`)
    will be created if it does not exist.
    """
    _f = produces_document(function, path)
    return _f(*args, **kwargs)


def produces_document(
    function: Callable[..., Any],
    path: Union[str, Path],
    check_args: bool = False,
) -> Callable[..., Any]:
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
    loads and returns `out/result.json`. Note that the parent directory of the
    output file (in this case, `out/`) will be created if it does not
    exist.However, if `out/result.json` exists and was produced by calling
    `_f(*args, **kwargs)`, then

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
    `out/result.json` but rather `out/result.json/<hash>.json` where `hash` is
    the MD5 hash of the serialization of the `args`/`kwargs` document above.
    """

    def _wrapped(path: Path, *args, **kwargs) -> Dict[str, Any]:
        if check_args:
            s = to_json({"args": args, "kwargs": kwargs})
            h = hashlib.md5(s.encode("utf-8")).hexdigest()
            path.mkdir(parents=True, exist_ok=True)
            path = path / f"{h}.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
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
