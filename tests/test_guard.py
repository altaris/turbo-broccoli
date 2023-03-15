# pylint: disable=missing-function-docstring
"""Test suite for `tdt.produces_document`"""

import json
from turbo_broccoli import guarded_call, produces_document


def test_guarded_call():
    def f(a: int):
        return {"a": a}

    path = "out/test_guarded_call.json"
    x = guarded_call(f, path, 1)
    with open(path, "r", encoding="utf-8") as fp:
        y = json.load(fp)
    assert isinstance(x, dict)
    assert x == y
    assert x != f(2)
    assert x == guarded_call(f, path, 2)  # Intended behavior


def test_produces_document():
    def f(a: int):
        return {"a": a}

    path = "out/test_produces_document.json"
    _f = produces_document(f, path, check_args=False)
    x = _f(1)
    with open(path, "r", encoding="utf-8") as fp:
        y = json.load(fp)
    assert isinstance(x, dict)
    assert x == y
    assert x != f(2)
    assert x == _f(2)  # Intended behavior


def test_produces_document_check_args():
    def f(a: int):
        return {"a": a}

    path = "out/test_produces_document_check_args.json"
    _f = produces_document(f, path, check_args=True)
    assert _f(1) == {"a": 1}
    assert _f(2) == {"a": 2}
    assert _f(1) == {"a": 1}  # Repetition intended
    assert _f(2) == {"a": 2}  # Repetition intended
    assert _f(1) != _f(2)
