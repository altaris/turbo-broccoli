"""scipy objects"""
__docformat__ = "google"

from typing import Any, Callable, List, Tuple

from scipy.sparse import csr_matrix


def _csr_matrix_to_json(m: csr_matrix) -> dict:
    """Converts a csr_matrix into a JSON document."""
    return {
        "__type__": "csr_matrix",
        "__version__": 1,
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
        1: _json_to_csr_matrix_v1,
    }
    return DECODERS[dct["__version__"]](dct)


def _json_to_csr_matrix_v1(dct: dict) -> csr_matrix:
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
    DECODERS = {
        "csr_matrix": _json_to_csr_matrix,
    }
    try:
        return DECODERS[dct["__scipy__"]["__type__"]](dct["__scipy__"])
    except KeyError as exc:
        raise TypeError("Not a valid scipy document") from exc


def to_json(obj: Any) -> dict:
    """
    Serializes a Scipy object into JSON by cases. See the README for the
    precise list of supported types.

    The return dict has the following structure

        {
            "__scipy__": {...},
        }

    where the `{...}` dict contains the actual data, and whose structure
    depends on the precise type of `obj`.
    """
    ENCODERS: List[Tuple[type, Callable[[Any], dict]]] = [
        (csr_matrix, _csr_matrix_to_json),
    ]
    for t, f in ENCODERS:
        if isinstance(obj, t):
            return {"__scipy__": f(obj)}
    raise TypeError("Not a supported scipy type")
