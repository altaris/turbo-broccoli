# pylint: disable=import-outside-toplevel
# pylint: disable=missing-function-docstring
# pylint: disable=unused-import
"""Test suite for guarded blocks/loops"""

from pathlib import Path

import common  # Must be before turbo_broccoli imports
from numpy import result_type

from turbo_broccoli import GuardedBlockHandler, load_json

TEST_PATH = Path("out") / "test"


def test_guarded_bloc_handler():
    path = TEST_PATH / "test_guarded_bloc_handler.json"
    h = GuardedBlockHandler(path)
    for _ in h:
        h.result = 41
        h.result = 42
    for _ in h:  # Block should be skipped
        assert False
    assert h.result == 42
    assert h.result == load_json(path)


def test_guarded_bloc_handler_iter_dict():
    path = TEST_PATH / "test_guarded_bloc_handler_iter_dict.json"
    h = GuardedBlockHandler(path)
    l = ["a", "b", "c", "d"]
    for i, x in h(l[:-1], result_type="dict"):
        h.result[x] = [i, x]
    for i, x in h(l, result_type="dict"):
        assert x == l[-1]  # a, b, c should be skipped
        h.result[x] = [i, x]
    assert h.result == {x: [i, x] for i, x in enumerate(l)}
    for i, x in h(l, result_type="dict"):
        assert False  # entire loop should be skipped


def test_guarded_bloc_handler_iter_list():
    path = TEST_PATH / "test_guarded_bloc_handler_iter_list.json"
    h = GuardedBlockHandler(path)
    l = ["a", "b", "c", "d"]
    for i, x in h(l[:-1], result_type="list"):
        h.result.append([i, x])
    for i, x in h(l, result_type="list"):
        assert x == l[-1]  # a, b, c should be skipped
        h.result.append([i, x])
    assert h.result == [[i, x] for i, x in enumerate(l)]
    for i, x in h(l, result_type="list"):
        assert False  # entire loop should be skipped


def test_guarded_bloc_handler_native():
    import pandas as pd

    path = TEST_PATH / "test_guarded_bloc_handler_native.csv"
    h = GuardedBlockHandler(path)
    df1 = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]})
    for _ in h:
        h.result = df1
    for _ in h:  # Block should be skipped
        assert False
    df2 = pd.read_csv(path)
    if "Unnamed: 0" in df2.columns:
        df2.drop(["Unnamed: 0"], axis=1, inplace=True)
    assert (df1 == df2).all().all()


def test_guarded_bloc_handler_no_load():
    path = TEST_PATH / "test_guarded_bloc_handler_no_load.json"
    h1 = GuardedBlockHandler(path)
    for _ in h1:
        h1.result = 42
    h2 = GuardedBlockHandler(path, load_if_skip=False)
    for _ in h2:
        h2.result = 41
    assert h1.result == load_json(path)
    assert h2.result is None
