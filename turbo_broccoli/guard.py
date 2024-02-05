"""Guarded call"""

from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from typing import Any, Generator

from turbo_broccoli.native import save, load
from turbo_broccoli.context import Context


class GuardedBlockHandler:
    """
    A guarded block handler allows to guard an entire block/loop of code. Use
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
    See `turbo_broccoli.native.save` and `turbo_broccoli.native.load`.
    """

    block_name: str | None
    context: Context
    result: Any = None

    def __init__(
        self, file_path: str | Path, block_name: str | None = None, **kwargs
    ) -> None:
        """
        Args:
            file_path (str | Path): Output file path
            block_name (str | None, optional): Name of the block, for logging
                purposes.
            **kwargs: Forwarded to the `turbo_broccoli.context.Context`
                constructor.
        """
        kwargs["file_path"] = file_path
        self.block_name, self.context = block_name, Context(**kwargs)

    def guard(self) -> Generator[Any, None, None]:
        """See `turbo_broccoli.guard.GuardedBlockHandler`'s documentation"""
        assert isinstance(self.context.file_path, Path)  # for typechecking
        if self.context.file_path.is_file():
            self.result = load(self.context.file_path)
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
                save(self.result, self.context.file_path)
                if self.block_name is not None:
                    logging.debug(
                        f"Saved guarded block '{self.block_name}' results to "
                        f"'{self.context.file_path}'"
                    )
