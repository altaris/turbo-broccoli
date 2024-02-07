# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
"""deque (de)serialization test suite"""

import numpy as np
from common import to_from_json
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix


def _assert_csr_matrix_equal(m1: csr_matrix, m2: csr_matrix) -> None:
    assert not (m1 != m2).toarray().any()
    assert_array_equal(m1.toarray(), m2.toarray())


def test_csr_matrix():
    m = csr_matrix((3, 4), dtype=np.int8)
    _assert_csr_matrix_equal(m, to_from_json(m))

    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    m = csr_matrix((data, (row, col)), shape=(3, 3))
    _assert_csr_matrix_equal(m, to_from_json(m))

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    m = csr_matrix((data, indices, indptr), shape=(3, 3))
    _assert_csr_matrix_equal(m, to_from_json(m))

    row = np.array([0, 1, 2, 0])
    col = np.array([0, 1, 1, 0])
    data = np.array([1, 2, 4, 8])
    m = csr_matrix((data, (row, col)), shape=(3, 3))
    _assert_csr_matrix_equal(m, to_from_json(m))
