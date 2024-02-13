"""Guarded block handler"""

from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from typing import Any, Generator

from turbo_broccoli.context import Context
from turbo_broccoli.native import load as native_load
from turbo_broccoli.native import save as native_save


class GuardedBlockHandler:
    """
    If a block of code produces a JSON file, and if it is not needed to rerun
    it that the output file exists, then a guarded block handler if an
    alternative to

    ```py
    if not Path("out/foo.json").exists():
        ...
        if success:
            tb.save_json(result, "out/foo.json")
    else:
        result = tb.load_json("out/foo.json")
    ```

    A guarded block handler allows to *guard* an entire block of code. Use
    it as follows:

    ```py
    h = GuardedBlockHandler("out/foo.json")
    for _ in h.guard():
        # This whole block will be skipped if out/foo.json exists
        # If not, don't forget to set the results:
        h.result = ...
    # In any case, the results of the block are available in h.result
    ```

    (I know the syntax isn't the prettiest. It would be more natural to use a
    `with h.guard():` syntax but python doesn't allow for context managers that
    don't yield...) The handler's `result` is `None` by default. If left to
    `None`, no output file is created. This allows for scenarios like

    ```py
    h = GuardedBlockHandler("out/foo.json")
    for _ in h.guard():
        ... # Guarded code
        if success:
            h.result = ...
    ```

    It is also possible to use "native" saving/loading methods:

    ```py
    h = GuardedBlockHandler("out/foo.csv")
    for _ in h.guard():
        ...
        h.result = some_pandas_dataframe
    ```

    See `turbo_broccoli.native.save` and `turbo_broccoli.native.load`. Finally,
    if the actual result of the block are not needed, use:

    ```py
    h = GuardedBlockHandler("out/large.json", load_if_skip=False)
    for _ in h.guard():
        ...
    # If the block was skipped (out/large.json already exists), h.result is
    # None instead of the content of out/large.json
    ```
    """

    block_name: str | None
    context: Context
    load_if_skip: bool
    result: Any = None

    def __init__(
        self,
        file_path: str | Path,
        block_name: str | None = None,
        load_if_skip: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            file_path (str | Path): Output file path.
            block_name (str, optional): Name of the block, for logging
                purposes. Can be left to `None` to suppress such logs.
            load_if_skip (bool, optional): Wether to load the output file if
                the block is skipped.
            **kwargs: Forwarded to the `turbo_broccoli.context.Context`
                constructor.
        """
        kwargs["file_path"] = file_path
        self.block_name, self.context = block_name, Context(**kwargs)
        self.load_if_skip = load_if_skip

    def guard(self) -> Generator[Any, None, None]:
        """See `turbo_broccoli.guard.GuardedBlockHandler`'s documentation"""
        assert isinstance(self.context.file_path, Path)  # for typechecking
        if self.context.file_path.is_file():
            if self.load_if_skip:
                self.result = native_load(self.context.file_path)
            if self.block_name:
                logging.debug(f"Skipped guarded block '{self.block_name}'")
        else:
            yield self
            if self.result is not None:
                assert isinstance(
                    self.context.file_path, Path
                )  # for typechecking
                self.context.file_path.parent.mkdir(
                    parents=True, exist_ok=True
                )
                native_save(self.result, self.context.file_path)
                if self.block_name is not None:
                    logging.debug(
                        f"Saved guarded block '{self.block_name}' results to "
                        f"'{self.context.file_path}'"
                    )
