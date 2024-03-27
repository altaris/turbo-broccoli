# pylint: disable=missing-function-docstring
"""Pytorch (de)serialization test suite"""

import torch
from common import to_from_json
from torch import Tensor
from torch.nn import Module
from torch.testing import assert_close
from torch.utils.data import (
    ConcatDataset,
    IterableDataset,
    StackDataset,
    Subset,
    TensorDataset,
)

from turbo_broccoli import Context


class TestModule(Module):
    """A test module to test things with pytorch modules"""

    module: torch.nn.Module

    def __init__(self):
        super().__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.module.forward(x)


def _assert_iterable_datasets_equal(
    ds1: IterableDataset, ds2: IterableDataset
):
    for a, b in zip(ds1, ds2):
        assert_close(a, b)


def _make_test_dataset() -> TensorDataset:
    return TensorDataset(
        torch.rand((8, 3)), torch.rand((8, 3)), torch.rand((8, 3))
    )


def test_pytorch_numerical():
    x = Tensor()
    assert to_from_json(x).numel() == 0
    x = Tensor([1, 2, 3])
    assert_close(x, to_from_json(x))
    x = torch.rand((10, 10))
    assert_close(x, to_from_json(x))


def test_pytorch_numerical_large():
    ctx = Context(min_artifact_size=0)
    x = torch.rand((100, 100), dtype=torch.float64)
    assert_close(x, to_from_json(x, ctx))


def test_pytorch_module():
    ctx = Context(pytorch_module_types=[TestModule])
    x = torch.ones(4)
    a = TestModule()
    b = to_from_json(a, ctx)
    assert_close(a(x), b(x))


def test_pytorch_concatdataset():
    x = ConcatDataset([_make_test_dataset(), _make_test_dataset()])
    y = to_from_json(x)
    assert isinstance(y, ConcatDataset)
    _assert_iterable_datasets_equal(x, y)


def test_pytorch_stackdataset_tuple():
    x = StackDataset(_make_test_dataset(), _make_test_dataset())
    y = to_from_json(x)
    assert isinstance(y, StackDataset)
    _assert_iterable_datasets_equal(x, y)


def test_pytorch_stackdataset_dict():
    x = StackDataset(u=_make_test_dataset(), v=_make_test_dataset())
    y = to_from_json(x)
    assert isinstance(y, StackDataset)
    _assert_iterable_datasets_equal(x, y)


def test_pytorch_subset():
    idx = [1, 3, 5]
    x = Subset(_make_test_dataset(), idx)
    y = to_from_json(x)
    assert isinstance(y, Subset)
    assert x.indices == y.indices
    _assert_iterable_datasets_equal(x, y)


def test_pytorch_tensordataset():
    x = _make_test_dataset()
    y = to_from_json(x)
    assert isinstance(y, TensorDataset)
    assert len(x.tensors) == len(y.tensors)
    for a, b in zip(x.tensors, y.tensors):
        assert_close(a, b)
