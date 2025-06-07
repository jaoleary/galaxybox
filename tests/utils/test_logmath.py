"""Unit tests for logadd and logsum functions in galaxybox.utils.logmath."""

import numpy as np
import pytest

from galaxybox.utils.logmath import logadd, logsum


@pytest.mark.parametrize(
    "input, base, expected",
    [
        ([9, 10, 11], 10, 11.04532),
        ([9, 10, 11], np.e, 11.40761),
        ([0, 0, 0], 10, 0.477121),  # log10(3)
        ([0, 0, 0], np.e, 1.098612),  # ln(3)
        ([1, 2, 3], 10, 3.46497),
        ([1, 2, 3], 2, 3.90689),
    ],
)
def test_logsum_with_different_bases(input, base, expected):
    result = logsum(input, base=base)
    np.testing.assert_almost_equal(result, expected, decimal=5)


@pytest.mark.parametrize(
    "a, b, base, expected",
    [
        ([9, 10, 11], [9, 10, 11], 10, np.array([9.30103, 10.30103, 11.30103])),
        ([9, 10, 11], [9, 10, 11], np.e, np.array([9.69315, 10.69315, 11.69315])),
        ([0, 0, 0], [0, 0, 0], 10, np.full(3, np.log10(2))),
        ([0, 1, 2], [2, 1, 0], 10, np.log10(10 ** np.array([0, 1, 2]) + 10 ** np.array([2, 1, 0]))),
        ([1, 2, 3], [4, 5, 6], 2, np.log2(2 ** np.array([1, 2, 3]) + 2 ** np.array([4, 5, 6]))),
    ],
)
def test_logadd_with_different_bases(a, b, base, expected):
    result = logadd(a, b, base=base)
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_logsum_empty_input():
    with pytest.raises(ValueError):
        logsum([], base=10)


def test_logadd_empty_input():
    with pytest.raises(ValueError):
        logadd([], [], base=10)


def test_logadd_shape_mismatch():
    with pytest.raises(ValueError):
        logadd([1, 2], [1], base=10)
