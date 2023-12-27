"""scipy objects"""
__docformat__ = "google"

from typing import Any, Callable, Tuple

from scipy.sparse import csr_matrix

from turbo_broccoli.utils import (
    DeserializationError,
    TypeNotSupported,
    raise_if_nodecode,
)


def _csr_matrix_to_json(m: csr_matrix) -> dict:
    """Converts a csr_matrix into a JSON document."""
    return {
        "__type__": "scipy.csr_matrix",
        "__version__": 2,
        "data": m.data,
        "dtype": m.dtype,
        "indices": m.indices,
        "indptr": m.indptr,
        "shape": m.shape,
    }


def _json_to_csr_matrix(dct: dict) -> csr_matrix:
    """
    Converts a JSON document to a csr_matrix. See `to_json` for the
    specification `dct` is expected to follow. Note that the key `__scipy__`
    should not be present.
    """
    DECODERS = {
        # 1: _json_to_csr_matrix_v1,  # Use turbo_broccoli v3
        2: _json_to_csr_matrix_v2,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_csr_matrix_v2(dct: dict) -> csr_matrix:
    """
    Converts a JSON document to a csr_matrix following the v1 specification.
    """
    return csr_matrix(
        (dct["data"], dct["indices"], dct["indptr"]),
        shape=dct["shape"],
        dtype=dct["dtype"],
    )


def from_json(dct: dict) -> Any:
    """
    Deserializes a dict into a csr_matrix. See `to_json` for the specification
    `dct` is expected to follow. In particular, note that `dct` must contain
    the key `__csr_matrix__`.
    """
    raise_if_nodecode("scipy")
    DECODERS = {
        "scipy.csr_matrix": _json_to_csr_matrix,
    }
    try:
        type_name = dct["__type__"]
        raise_if_nodecode(type_name)
        return DECODERS[type_name](dct)
    except KeyError as exc:
        raise DeserializationError() from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a Scipy object into JSON by cases. See the README for the
    precise list of supported types. The return dict has the following
    structure:

    - [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)

            {
                "__type__": "scipy.csr_matrix",
                "__version__": 2,
                "data": ...,
                "dtype": ...,
                "indices": ...,
                "indptr": ...,
                "shape": ...,
            }

    """
    ENCODERS: list[Tuple[type, Callable[[Any], dict]]] = [
        (csr_matrix, _csr_matrix_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return f(obj)
    raise TypeNotSupported()
