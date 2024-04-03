# pylint: disable=missing-function-docstring
# pylint: disable=unnecessary-lambda-assignment
"""Parallel-guard test suite"""

from pathlib import Path

import pytest

from turbo_broccoli import Parallel, delayed
from turbo_broccoli.turbo_broccoli import load_json

TEST_PATH = Path("out") / "test"


def _get_path(name: str) -> Path:
    """Creates `TEST_PATH/name.json` and deletes it if it already exists"""
    path = TEST_PATH / (name + ".json")
    if path.exists():
        path.unlink()
    return path


def test_parallel_one_arg_1():
    path = _get_path("test_parallel_one_arg_1")
    f = lambda x: x + x
    lst = [str(i) for i in range(10)]
    jobs = [delayed(f)(x) for x in lst]
    results = Parallel(path, n_jobs=1, only_one_arg=True)(jobs)
    assert results == {x: f(x) for x in lst}
    assert results == load_json(path)


def test_parallel_one_arg_2():
    path = _get_path("test_parallel_one_arg_2")
    f = lambda x: x + x
    g = lambda x: x + x + x
    lst = [str(i) for i in range(4)]
    jobs = [delayed(f)(x) for x in lst[:2]]
    results = Parallel(path, n_jobs=1, only_one_arg=True)(jobs)
    jobs = [delayed(g)(x) for x in lst]
    results = Parallel(path, n_jobs=1, only_one_arg=True)(jobs)
    assert results == {"0": "00", "1": "11", "2": "222", "3": "333"}
    assert results == load_json(path)


def test_parallel_one_arg_3():
    path = _get_path("test_parallel_one_arg_3")
    f = lambda x: x + x
    lst = [str(i) for i in range(1000)]
    jobs = [delayed(f)(x) for x in lst]
    results = Parallel(path, n_jobs=2, only_one_arg=True)(jobs)
    assert list(results.keys()) == lst


def test_parallel_1():
    path = _get_path("test_parallel_1")
    f = lambda a, b: a * b
    lst = [(str(i), i) for i in range(3)]
    jobs = [delayed(f)(*x) for x in lst]
    results = Parallel(path, n_jobs=1, only_one_arg=False)(jobs)
    assert results == {x: f(*x) for x in lst}
    assert results == load_json(path)


def test_parallel_2():
    path = _get_path("test_parallel_2")
    f = lambda a, b: a * b
    g = lambda a, b: a * (b + 1)
    lst = [(str(i), i) for i in range(4)]
    jobs = [delayed(f)(*x) for x in lst[:2]]
    results = Parallel(path, n_jobs=1, only_one_arg=False)(jobs)
    jobs = [delayed(g)(*x) for x in lst]
    results = Parallel(path, n_jobs=1, only_one_arg=False)(jobs)
    assert results == {
        ("0", 0): "",
        ("1", 1): "1",
        ("2", 2): "222",
        ("3", 3): "3333",
    }
    assert results == load_json(path)


def test_parallel_sanity_same_args_1():
    path = _get_path("test_parallel_sanity_same_args_1")
    f = lambda a: 2 * a
    lst = [1, 2, 3, 1]
    jobs = [delayed(f)(x) for x in lst]
    with pytest.raises(ValueError):
        Parallel(path, n_jobs=1, only_one_arg=True)(jobs)


def test_parallel_sanity_same_args_2():
    path = _get_path("test_parallel_sanity_same_args_2")
    f = lambda a, b: a * b
    lst = [(0, 0), (0, 1), (1, 1), (0, 0)]
    jobs = [delayed(f)(*x) for x in lst]
    with pytest.raises(ValueError):
        Parallel(path, n_jobs=1, only_one_arg=False)(jobs)


def test_parallel_sanity_only_one_arg():
    path = _get_path("test_parallel_sanity_only_one_arg")
    f = lambda a, b: a * b
    lst = [(0, 0), (0, 1), (1, 1)]
    jobs = [delayed(f)(*x) for x in lst]
    with pytest.raises(ValueError):
        Parallel(path, n_jobs=1, only_one_arg=True)(jobs)


def test_parallel_sanity_kwargs():
    def f(x):
        return x + x

    with pytest.raises(ValueError):
        # pylint: disable=expression-not-assigned
        [delayed(f)(x=x) for x in range(5)]
