# pylint: disable=missing-function-docstring
"""Test suite for `tdt.produces_document`"""

import json
from turbo_broccoli import produces_document


def test_produces_document():
    def f(a: int):
        return {"a": a}

    path = "out/test_produces_document.json"
    _f = produces_document(f, path)
    x = _f(1)
    with open(path, "r", encoding="utf-8") as fp:
        y = json.load(fp)
    assert isinstance(x, dict)
    assert x == y
    assert x != f(2)
    assert x == _f(2)
