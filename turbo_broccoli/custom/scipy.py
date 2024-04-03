"""scipy objects"""

from typing import Any, Callable, Tuple

from scipy.sparse import csr_matrix

from turbo_broccoli.context import Context
from turbo_broccoli.exceptions import DeserializationError, TypeNotSupported


def _csr_matrix_to_json(m: csr_matrix, ctx: Context) -> dict:
    return {
        "__type__": "scipy.csr_matrix",
        "__version__": 2,
        "data": m.data,
        "dtype": m.dtype,
        "indices": m.indices,
        "indptr": m.indptr,
        "shape": m.shape,
    }


def _json_to_csr_matrix(dct: dict, ctx: Context) -> csr_matrix:
    DECODERS = {
        2: _json_to_csr_matrix_v2,
    }
    return DECODERS[dct["__version__"]](dct, ctx)


def _json_to_csr_matrix_v2(dct: dict, ctx: Context) -> csr_matrix:
    return csr_matrix(
        (dct["data"], dct["indices"], dct["indptr"]),
        shape=dct["shape"],
        dtype=dct["dtype"],
    )


# pylint: disable=missing-function-docstring
def from_json(dct: dict, ctx: Context) -> Any:
    DECODERS = {
        "scipy.csr_matrix": _json_to_csr_matrix,
    }
    try:
        type_name = dct["__type__"]
        return DECODERS[type_name](dct, ctx)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any, ctx: Context) -> dict:
    """
    Serializes a Scipy object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)

        ```py
        {
            "__type__": "scipy.csr_matrix",
            "__version__": 2,
            "data": ...,
            "dtype": ...,
            "indices": ...,
            "indptr": ...,
            "shape": ...,
        }
        ```

    """
    ENCODERS: list[Tuple[type, Callable[[Any, Context], dict]]] = [
        (csr_matrix, _csr_matrix_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj, ctx)
    raise TypeNotSupported()
