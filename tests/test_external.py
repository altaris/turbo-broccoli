"""deque (de)serialization test suite"""

from pathlib import Path

import numpy as np
import pytest
from common import to_from_json
from numpy.testing import assert_almost_equal

from turbo_broccoli import Context, ExternalData, save
from turbo_broccoli.turbo_broccoli import load_json, save_json

TEST_PATH = "out/test/test_external/"


def test_external_no_path():
    q = TEST_PATH + "test_external_no_path/a.np"
    with pytest.raises(ValueError):
        ExternalData(q, Context())


def test_external_relative_1():
    p = TEST_PATH + "test_external_relative_1.json"
    q = "test_external_relative_1/a.np"
    ctx = Context(p)
    a = np.arange(10)
    save(a, TEST_PATH + q)
    x = {"a": ExternalData(q, ctx)}
    assert_almost_equal(a, x["a"].data)
    y = to_from_json(x, ctx)
    assert_almost_equal(a, y["a"].data)


def test_external_relative_2():
    p = TEST_PATH + "test_external_relative_2.json"
    q = "test_external_relative_2.np"
    ctx = Context(p)
    a = np.arange(10)
    save(a, TEST_PATH + q)
    x = {"a": ExternalData(q, ctx)}
    assert_almost_equal(a, x["a"].data)
    y = to_from_json(x, ctx)
    assert_almost_equal(a, y["a"].data)


def test_external_absolute_1():
    p = Path(TEST_PATH + "test_external_absolute_1/a.json").absolute()
    q = Path(TEST_PATH + "test_external_absolute_1/x/y/z/a.np").absolute()
    ctx = Context(p)
    a = np.arange(10)
    save(a, q)
    x = {"a": ExternalData(q, ctx)}
    assert_almost_equal(a, x["a"].data)
    y = to_from_json(x, ctx)
    assert_almost_equal(a, y["a"].data)


def test_external_absolute_2():
    p = Path(TEST_PATH + "test_external_absolute_2/a.json").absolute()
    q = Path(TEST_PATH + "test_external_absolute_2/a.np").absolute()
    ctx = Context(p)
    a = np.arange(10)
    save(a, q)
    x = {"a": ExternalData(q, ctx)}
    assert_almost_equal(a, x["a"].data)
    y = to_from_json(x, ctx)
    assert_almost_equal(a, y["a"].data)


def test_external_nodecode_path():
    p = TEST_PATH + "test_external_nodecode_path.json"
    q = "test_external_nodecode_path/a.np"
    ctx = Context(p)
    a = np.arange(10)
    save(a, TEST_PATH + q)
    x = {"a": ExternalData(q, ctx)}
    save_json(x, ctx.file_path, ctx)
    y = load_json(ctx.file_path, nodecode_types=["pathlib"])
    assert isinstance(y["a"], dict)
