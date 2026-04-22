# -*- coding: utf-8 -*-

"""utils.py

Utility functions for the ocnus python package.
"""

import numpy as np
import scipy as sp


def discretize_array(timestamps, t_start, t_end, t_count, array):
    """Discretize a single or multiple data arrays.

    Args:
        timestamps: The input timestamps.
        t_start: The start time of the discretization.
        t_end: The end time of the discretization.
        t_count: The number of discrete time steps.
        array: The input data array(s) to be discretized. Must be 1D or 2D numpy array(s).

    Returns:
        The discrete time stamps and corresponding data arrays.

    Raises:
        ValueError: If the start or end timestamps are invalid.
    """
    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)

    if not (t_start <= timestamps[0] < timestamps[-1] <= t_end):
        raise ValueError("Invalid time range")
    elif not all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("Timestamps must be strictly increasing")

    ts_discr = np.linspace(t_start, t_end, t_count, endpoint=True)

    if isinstance(array, list):
        return ts_discr, [
            discretize_array(timestamps, t_start, t_end, t_count, arr)[1]
            for arr in array
        ]
    else:
        if len(array.shape) == 1:
            data = array[:, np.newaxis]
        else:
            data = array

        interp = np.zeros((t_count, data.shape[1]))

        for idx in range(data.shape[1]):
            b_spline = sp.interpolate.make_lsq_spline(
                timestamps,
                np.nan_to_num(data[:, idx]),
                np.array(3 * [timestamps[0]] + list(ts_discr) + 3 * [timestamps[-1]]),
                w=~np.isnan(data[:, idx]),
            )

            interp[:, idx] = b_spline(ts_discr)

        return ts_discr, interp
