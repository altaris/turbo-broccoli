"""
Guarded parallel calls

## Usage

It works in a way that is similar to
[`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html),
for example

```py
from math import sqrt
import turbo_broccoli as tb

# Note the use of `tb.delayed` instead of `joblib.delayed`.
#          ↓
jobs = [tb.delayed(sqrt)(i**2) for i in range(5)]
executor = tb.Parallel("foo/bar.json", only_one_arg=True, n_jobs=2)
results = executor(jobs)
```

gives

```py
{0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}
```

however, only the calls for which the corresponding entry in `out/foo.json`
does not exist will actually be executed, the others will simply be loaded.
Note that unlike `joblib.Parallel`, the result is a `dict` of arg/result, not
just a `list` of results.

If the function in the jobs take more than one argument, simply drop the
`only_one_arg=True` argument:

```py
from math import sqrt
import turbo_broccoli as tb

f = lambda a, b: a * b

jobs = [tb.delayed(f)(i, j) for i in range(5) for j in range(5)]
executor = tb.Parallel("foo/bar.json", n_jobs=2)
results = executor(jobs)
```

gives

```py
{(0, 0): 0, (0, 1): 0, ..., (4, 4): 16}
```

## Notes & caveats

* The result of `executor(jobs)` is a dict or a generator of key/value pairs.

* The order of the results is guaranteed to be consistent with the order of the
  jobs.

* The argument(s) in the jobs must be hashable. Furthermore, if a job has
  kwargs, a `ValueError` is raised.

* Every job must have a different tuple of argument, or in other words, every
  job must be unique. So something like this is not acceptable:
    ```py
    f = lambda a: 2 * a

    jobs = [tb.delayed(f)(i % 2) for i in range(5)]
    executor = tb.Parallel(...)
    results = executor(jobs)
    ```
    because `f` is in effect multiple times with value `0`. This is not checked
    because I'm too lazy but I promise you it'll fuck shit up. In particular,
    TurboBroccoli's `Parallel` is not suited for functions with no arguments
    (unless if they are executed only once but that kind of defeats the idea of
    parallelism).

* Beyond the arguments documented in `Parallel`, [`joblib.Parallel`
  arguments](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
  can be passed as kwargs to the constructor.

* TurboBroccoli's `Parallel` honors the `return_as` argument of
  [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
  (which can be `"list"` or `"generator"`). However, the value
  `return_as="generator_unordered"` is not supported and will fall back to
  `"generator"` with a warning. Also, note that eventhough you might set
  `return_as="list"`, the result will still be a dict.

* Everytime a new result is obtained, it is immediately written to the output
  file. This means that if there are N jobs, the output file will be written to
  up to N times. If the results are accessed in quick succession (e.g.
  `return_as="list"` which is the default), this can slam the filesystem pretty
  hard.

* If the output of the job's function are large, it might be inefficient to
  rewrite every past results the output file each time a new result is
  generated. Using [embedded
  objects](https://altaris.github.io/turbo-broccoli/turbo_broccoli/custom/embedded.html),
  can make writes faster.

    ```py
    def f(i):
        ...
        return tb.EmbeddedDict({"result": something_big, ...})

    jobs = [tb.delayed(f)(i) for i in range(1000)]
    executor = tb.Parallel(...)
    results = executor(jobs)
    ```
"""

from pathlib import Path
from typing import Any, Callable, Generator, Iterable

import joblib

try:
    from loguru import logger as logging
except ModuleNotFoundError:
    import logging  # type: ignore

from .context import Context
from .turbo_broccoli import load_json, save_json


class _DelayedCall:

    function: Callable
    args: tuple[Any, ...]

    def __call__(self, *args: Any, **kwds: Any) -> "_DelayedCall":
        if kwds:
            raise ValueError("Keyword arguments are not supported")
        self.args = args
        return self

    def __init__(self, function: Callable) -> None:
        self.function = function

    def to_joblib_delayed(self) -> Callable:
        """
        Returns a `joblib.delayed` object that can be used with
        `joblib.Parallel`.
        """
        return joblib.delayed(self.function)(*self.args)


class Parallel:
    """
    Guarded analogue to
    [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html).
    See module documentation.
    """

    context: Context
    executor: joblib.Parallel
    lonly_one_arg: bool

    def __init__(
        self,
        output_file: str | Path,
        context: Context | None = None,
        only_one_arg: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            output_file (str | Path):
            context (Context | None, optional):
            only_one_arg (bool, optional): If `True`, assumes that every job
                has exactly one argument. This produces more compact output
                files.
            kwargs (Any): Forwarded to
                [`joblib.Parallel`](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)
        """
        if kwargs.get("return_as") == "generator_unordered":
            logging.warning(
                "The option return_as='generator_unordered' is not supported. "
                "Using 'generator' instead."
            )
            kwargs["return_as"] = "generator"
        self.context = context or Context(output_file)
        self.executor = joblib.Parallel(**kwargs)
        self.only_one_arg = only_one_arg

    def __call__(
        self, jobs: Iterable[_DelayedCall]
    ) -> dict | Generator[tuple[Any, Any], None, None]:
        g = self._call(jobs)
        if self.executor.return_generator:
            return g
        return dict(g)

    # pylint: disable=stop-iteration-return
    def _call(
        self, jobs: Iterable[_DelayedCall]
    ) -> Generator[tuple[Any, Any], None, None]:
        """
        Executes the jobs in parallel and yields the results. Saves to the
        output file each time a new result (i.e. one that was not already
        present in the output file) is obtained.

        Args:
            jobs (Iterable[_DelayedCall]): All the jobs, including those whose
                results are already in the output file (and therefore shall not
                be run again)
        """

        def _key(j: _DelayedCall) -> Any:
            """What the key of a job should be in the result dict"""
            return j.args[0] if self.only_one_arg else tuple(j.args)

        job_status = {
            _key(j): {"job": j, "done": False, "result": None} for j in jobs
        }

        # Check if some jobs already have their results in the output file
        assert self.context.file_path is not None  # for typechecking
        if self.context.file_path.exists():
            results = load_json(self.context.file_path, self.context)
            if not isinstance(results, dict):
                raise RuntimeError(
                    f"The contents of '{self.context.file_path}' is not a dict"
                )
            # Mark the jobs that are already done
            for k, r in results.items():
                if k in job_status:
                    job_status[k]["done"], job_status[k]["result"] = True, r
        else:
            results = {}

        new_results_it = iter(
            self.executor(
                d["job"].to_joblib_delayed()  # type: ignore
                for d in job_status.values()
                if not d["done"]
            )
        )
        # This loops strongly assumes that `jobs`, `loaded_results`,
        # `job_status`, and `new_results` are ordered in a consistent way
        for k, s in job_status.items():
            if s["done"]:
                yield k, s["result"]
            else:
                results[k] = next(new_results_it)
                save_json(results, self.context.file_path, self.context)
                yield k, results[k]

        # At this point, new_results should have been fully consumed
        try:
            next(new_results_it)
            raise RuntimeError("The executor returned too many results")
        except StopIteration:
            pass


def delayed(function: Callable) -> Callable:
    """Use this like `joblib.delayed`"""
    return _DelayedCall(function)
