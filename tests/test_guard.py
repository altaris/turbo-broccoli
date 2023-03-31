# pylint: disable=missing-function-docstring
"""Test suite for `tdt.produces_document`"""

from turbo_broccoli import (
    GuardedBlockHandler,
    guarded_call,
    produces_document,
    load_json,
)


def test_guarded_bloc_handler():
    path = "out/test_guarded_bloc_handler.json"
    h = GuardedBlockHandler(path)
    for _ in h.guard():
        h.result = 41
        h.result = 42
    for _ in h.guard():
        h.result = 43
    y = load_json(path)
    assert h.result == 42
    assert h.result == y


def test_guarded_call():
    def f(a: int):
        return {"a": a}

    path = "out/test_guarded_call.json"
    x = guarded_call(f, path, 1)
    y = load_json(path)
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
    y = load_json(path)
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
