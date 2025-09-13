"""Tools for performing basic operation on log values."""

from typing import Sequence, Union

import numpy as np
from scipy.special import logsumexp


def logsum(
    x: Sequence[int | float],
    base: int | float = 10,
) -> np.ndarray:
    """Compute the logarithm of the sum of exponentials.

    Parameters
    ----------
    x : Sequence[int | float]
        The input sequence of numbers for which the log-sum-exp is computed.
    base : int | float, optional
        The base for the logarithm calculation. Default is 10.

    Returns
    -------
    np.ndarray
        The computed log-sum-exp of the input array `x` in the specified logarithm `base`.

    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Input array cannot be empty")
    factor = np.log(base)
    return logsumexp(x * factor) / factor


def logadd(
    a: Sequence[Union[int, float]],
    b: Sequence[Union[int, float]],
    base: Union[int, float] = 10,
) -> np.ndarray:
    """Compute the logarithm of the sum of exponentials of two input arrays.

    Parameters
    ----------
    a : Sequence[Union[int, float]]
        The first input sequence of numbers.
    b : Sequence[Union[int, float]]
        The second input sequence of numbers.
    base : Union[int, float], optional
        The base for the logarithm calculation. Default is 10.

    Returns
    -------
    np.ndarray
        The computed log-sum-exp of the input arrays `a` and `b` in the specified logarithm `base`.

    """
    a = np.asarray(a)
    b = np.asarray(b)

    if a.size == 0 or b.size == 0:
        raise ValueError("Input array cannot be empty")

    if a.shape != b.shape:
        raise ValueError("Input arrays must have the same shape")

    factor = np.log(base)
    return np.logaddexp(a * factor, b * factor) / factor
