# pylint: disable=missing-function-docstring
"""Pytorch (de)serialization test suite"""

import os
import torch

from common import from_json, to_json


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
