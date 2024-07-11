# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np


def rng_truncanted(
    rng_func: Callable, maxv: float, minv: float, size: int, iter_max: int, **kwargs
) -> np.ndarray:
    """Truncated random number generator.

    Args:
        rng_func (Callable): random number generating function, must implement size argument
        maxv (float): random number ceiling
        minv (float): random number floor
        size (int): random numbers generated per iteration
        iter_max (int): maximum number of iterations allowed

    Raises:
        RuntimeError: if the number of maximum iterations is exceeded before (size) valid numbers are found

    Returns:
        np.ndarray: random numbers within given range
    """
    numbers = rng_func(size=size, **kwargs)

    for _ in range(iter_max):
        flt = (numbers > maxv) | (numbers < minv)
        if np.sum(flt) == 0:
            return numbers
        numbers[flt] = rng_func(size=len(flt), **kwargs)[flt]

    raise RuntimeError(
        "generating numbers inefficiently (%i/%i after %i iterations)",
        len(flt),
        size,
        iter_max,
    )
