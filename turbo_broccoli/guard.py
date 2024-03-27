"""
Guarded block handler

If a block of code produces a JSON file, say `out/foo.json`, and if it is not
needed to rerun the block if the output file exists, then a guarded block
handler if an alternative to

```py
if not Path("out/foo.json").exists():
    ...
    if success:
        tb.save_json(result, "out/foo.json")
else:
    result = tb.load_json("out/foo.json")
```

A guarded block handler allows to *guard* an entire block of code, and even a
loop on a per-iteration basis.

## Guarding a simple block

Use it as follows:

```py
h = GuardedBlockHandler("out/foo.json")
for _ in h:
    # This whole block will be skipped if out/foo.json exists
    # If not, don't forget to set the results:
    h.result = ...
# In any case, the results of the block are available in h.result
```

I know the syntax isn't the prettiest, it would be more natural to use a `with
h:` syntax but python doesn't allow for context managers that don't yield...
The handler's `result` is `None` by default. If `h.result` is left to `None`,
no output file is created. This allows for scenarios like

```py
h = GuardedBlockHandler("out/foo.json")
for _ in h:
    ...  # Guarded code
    if success:
        h.result = ...
```

It is also possible to use ["native" saving/loading
methods](https://altaris.github.io/turbo-broccoli/turbo_broccoli/native.html#save):

```py
h = GuardedBlockHandler("out/foo.csv")
for _ in h:
    ...
    h.result = some_pandas_dataframe
```

See `turbo_broccoli.native.save` and `turbo_broccoli.native.load`. Finally, if
the actual result of the block are not needed, use:

```py
h = GuardedBlockHandler("out/large.json", load_if_skip=False)
for _ in h:
    ...
# If the block was skipped (out/large.json already exists), h.result is
# None instead of the content of out/large.json
```

## Guarding a loop

Let's say you have a loop

```py
for x in an_iterable:
    ...  # expensive code that produces a result you want to save
```

You can guard the loop as follows:

```py
h = GuardedBlockHandler("out/foo.json")
for i, x in h(an_iterable):  # an_iterable is always enumerated!
    # h.result is already a dict, no need to initialize it
    ...  # expensive code that produces a result you want to save
    h.result[x] = ...
```

The contents of `h.result` are saved to `out/foo.json` at the end of every
iteration. However, if `out/foo.json` already exists, the loop will skip all
iterations that are already saved. In details, let's say that the contents of
`out/foo.json` is

```json
{"a": "aaa", "b": "bbb"}
```

Then the content of the following loop is only executed for `"c"`:

```py
for i, x in h(["a", "b", "c"]):
    h.result[x] = x * 3
# h.result is now {"a": "aaa", "b": "bbb", "c": "ccc"}
```

If you want `h.result` to be a list instead of a dict, use:

```py
h = GuardedBlockHandler("out/foo.json")
for i, x in h(an_iterable, result_type="list"):
    # h.result is already a list, no need to initialize it
    ...  # expensive code that produces a result you want to save
    h.result.append(...)
```

### Caveats

- Recall that in the case of simple blocks, setting/leaving `h.result` to
  `None` is understood as a failed computation:
  ```py
  for _ in h:
    h.result = None
  for _ in h:  # This block isn't skipped
    h.result = "Hello world"
  ```
  In the case of loops however, if an entry of `h.result` is set to `None`, the
  corresponding iteration is not treated as failed. For example:
  ```py
  for i, x in h(["a", "b", "c"]):
      h.result[x] = x * 3 if x != "b" else None
  # h.result is now {"a": "aaa", "b": None, "c": "ccc"}
  for i, x in h(["a", "b", "c"]):
      h.result[x] = x * 3
  # The second loop has been completely skipped, h.result is still
  # {"a": "aaa", "b": None, "c": "ccc"}
  ```
- The `load_if_skip` constructor argument has no effect, meaning that the JSON
  file is always loaded if it exists. If you want some level of laziness,
  consider the following trick:
  ```py
  from turbo_broccoli.context import EmbeddedDict

  h = GuardedBlockHandler("out/foo.json", nodecode_types=["embedded"])
  for i, x in h(["a", "b", "c"]):
    y = ...  # a dict that is expensive to compute
    h.result[x] = EmbeddedDict(y)
  ```
  By changing the type of `y` from a dict to an `EmbeddedDict`, and setting the
  `"embedded"` type in the guarded block handler's internal context's
  `nodecode_types`, results that were already present in the JSON file will not
  be decoded.
"""

from pathlib import Path

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from typing import Any, Generator, Iterable, Literal

from turbo_broccoli.context import Context
from turbo_broccoli.native import load as native_load
from turbo_broccoli.native import save as native_save


class GuardedBlockHandler:
    """See module documentation"""

    block_name: str | None
    context: Context
    file_path: Path
    load_if_skip: bool
    result: Any = None

    def __call__(
        self, it: Iterable, **kwargs
    ) -> Generator[tuple[int, Any], None, None]:
        """Alias for `GuardedBlockHandler.guard` with an iterable"""
        yield from self.guard(it, **kwargs)

    def __init__(
        self,
        file_path: str | Path,
        block_name: str | None = None,
        load_if_skip: bool = True,
        context: Context | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            file_path (str | Path): Output file path.
            block_name (str, optional): Name of the block, for logging
                purposes. Can be left to `None` to suppress such logs.
            load_if_skip (bool, optional): Wether to load the output file if
                the block is skipped.
            context (turbo_broccoli.context.Context, optional): Context to use
                when saving/loading the target JSON file. If left to `None`, a
                new context is built from the kwargs.
            **kwargs: Forwarded to the `turbo_broccoli.context.Context`
                constructor. Ignored if `context` is not `None`.
        """
        self.file_path = kwargs["file_path"] = Path(file_path)
        self.block_name, self.load_if_skip = block_name, load_if_skip
        self.context = context if context is not None else Context(**kwargs)

    def __iter__(self) -> Generator[Any, None, None]:
        """
        Alias for `GuardedBlockHandler.guard` with no iterable and no kwargs
        """
        yield from self.guard()

    def _guard_iter(
        self,
        it: Iterable,
        result_type: Literal["dict", "list"] = "dict",
        **kwargs,
    ) -> Generator[tuple[int, Any], None, None]:
        if self.file_path.is_file():
            self.result = native_load(self.file_path)
        else:
            self.result = {} if result_type == "dict" else []
        if result_type == "dict":
            yield from self._guard_iter_dict(it, **kwargs)
        else:
            yield from self._guard_iter_list(it, **kwargs)

    def _guard_iter_dict(
        self, it: Iterable, **kwargs
    ) -> Generator[tuple[int, Any], None, None]:
        for i, x in enumerate(it):
            if x in self.result:
                if self.block_name:
                    logging.debug(
                        f"Skipped iteration '{str(x)}' of guarded loop "
                        f"'{self.block_name}'"
                    )
                continue
            yield (i, x)
            self._save()

    def _guard_iter_list(
        self, it: Iterable, **kwargs
    ) -> Generator[tuple[int, Any], None, None]:
        for i, x in enumerate(it):
            if i < len(self.result):
                if self.block_name:
                    logging.debug(
                        f"Skipped iteration {i} of guarded loop "
                        f"'{self.block_name}'"
                    )
                continue
            yield (i, x)
            self._save()

    def _guard_no_iter(self, **kwargs) -> Generator[Any, None, None]:
        if self.file_path.is_file():
            self.result = (
                native_load(self.file_path) if self.load_if_skip else None
            )
            if self.block_name:
                logging.debug(f"Skipped guarded block '{self.block_name}'")
            return
        yield self
        if self.result is not None:
            self._save()
            if self.block_name is not None:
                logging.debug(
                    f"Saved guarded block '{self.block_name}' results to "
                    f"'{self.file_path}'"
                )

    def _save(self):
        """Saves `self.result`"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        native_save(self.result, self.file_path)

    def guard(
        self, it: Iterable | None = None, **kwargs
    ) -> Generator[Any, None, None]:
        """See `turbo_broccoli.guard.GuardedBlockHandler`'s documentation"""
        if it is None:
            yield from self._guard_no_iter(**kwargs)
        else:
            yield from self._guard_iter(it, **kwargs)
