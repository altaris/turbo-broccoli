# pylint: disable=missing-function-docstring
"""Pytorch (de)serialization test suite"""

import os
import torch

from turbo_broccoli.environment import register_pytorch_module_type

from common import from_json, to_json


class _TestModule(torch.nn.Module):

    module: torch.nn.Module

    def __init__(self):
        super(_TestModule, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Linear(4, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.module.forward(x)


def test_pytorch_numerical():
    x = torch.Tensor()
    assert from_json(to_json(x)).numel() == 0
    x = torch.Tensor([1, 2, 3])
    torch.testing.assert_close(x, from_json(to_json(x)))
    x = torch.rand((10, 10))
    torch.testing.assert_close(x, from_json(to_json(x)))


def test_pytorch_numerical_large():
    os.environ["TB_MAX_NBYTES"] = "0"
    x = torch.rand((100, 100), dtype=torch.float64)
    torch.testing.assert_close(x, from_json(to_json(x)))


def test_pytorch_module():
    register_pytorch_module_type(_TestModule)
    x = torch.ones(4)
    a = _TestModule()
    b = from_json(to_json(a))
    torch.testing.assert_close(a(x), b(x))
