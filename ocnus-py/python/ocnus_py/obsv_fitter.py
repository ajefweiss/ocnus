# -*- coding: utf-8 -*-

"""obsv_fitter.py

Implements the ObserVecFitter class for a simplified fitting approach with models that have ObserVec observables.
"""

import datetime as dt
import logging as lg
import numpy as np
import scipy as sp

from ocnus_py import MagObser, MagScObs, ObserVecNoise, Obser, ScObs


class ObserVecFitter:
    def __init__(
        self,
        model,
        t_start,
        t_end,
        ts,
        bd,
        tj,
        discretization=5,
        discretization_offset=1800,
        metric="rmse",
        threshold=1.0,
        nan_bounds=True,
        **kwargs
    ):
        """Initialize a ObserVecFitter object.

        Args:
            model: A model object.
            t_start: Starting time for the MO object, must be contained within ```ts```.
            t_end: End time for the MO object, must be contained within ```ts```.
            ts: Time array with either timestamps or datetime objects.
            bd: Data array for the corresponding time array.
            tj: Trajectory array
            discretization (int, optional): Number of discretization points. Must be at least 3. Defaults to 5.
            discretization_offset (float, optional): Time offset for the first and last discretization point (in seconds).
                This setting is only relevant if ```nan_bounds=True```.
                Must be at least 300 seconds. Defaults to 1800.
            nan_bounds: Flag that decides whether to set the first and last observation to NaN after offsetting. Defaults to True.
            metric (str, optional): The error metric by name. Defaults to "rmse".
            threshold (float, optional): Initial metric threshold used for initialization. Defaults to 1.0.

        Raises:
            ValueError: If the number of discretization points is less than 3.
            ValueError: If the discretization offset is smaller than 300 (5 minutes).
            ValueError: If the `ts` array does not contain either all timestamps or datetime objects.
            ValueError: If the ObserVec type has an unsupported number of dimensions.
            ValueError: If the metric is unsupported.
        """
        self.discr = discretization
        self.discr_offset = discretization_offset

        if self.discr < 3:
            raise ValueError(
                "number of discretization points must be larger than, or equal to, 3"
            )
        if self.discr_offset < 300:
            raise ValueError(
                "discretization offset  must be larger than, or equal to, 300 (5 minutes)"
            )

        # transform any datetime objects to timestamps
        if all(isinstance(_, dt.datetime) for _ in ts):
            ts = [_.timestamp() for _ in ts]
        elif all(isinstance(_, float) for _ in ts):
            pass
        else:
            raise ValueError(
                "ts argument must be either a list of datetimes or timestamps"
            )

        if isinstance(t_start, dt.datetime):
            t_start = t_start.timestamp()
        elif isinstance(t_start, float):
            pass
        else:
            raise ValueError("t_start argument must be either a datetime or timestamp")

        if isinstance(t_end, dt.datetime):
            t_end = t_end.timestamp()
        elif isinstance(t_end, float):
            pass
        else:
            raise ValueError("t_end argument must be either a datetime or timestamp")

        assert (
            t_start > ts[0] - self.discr_offset
        ), "MO start time is not within given time range"

        assert (
            t_end < ts[-1] + self.discr_offset
        ), "MO end time is not within given time range"

        self.duration = t_start, t_end

        if nan_bounds:
            self.ts_discr = np.array(
                [0] + list(np.linspace(t_start, t_end, self.discr, endpoint=True)) + [0]
            )

            self.ts_discr[0] = self.ts_discr[1] - self.discr_offset
            self.ts_discr[-1] = self.ts_discr[-2] + self.discr_offset

            self.bd_discr = np.zeros((self.discr + 2, bd.shape[1]))
            self.tj_discr = np.zeros((self.discr + 2, bd.shape[1]))

            for idx in range(bd.shape[1]):
                b_spline = sp.interpolate.make_lsq_spline(
                    ts,
                    np.nan_to_num(bd[:, idx]),
                    np.array(4 * [ts[0]] + list(self.ts_discr) + 4 * [ts[-1]]),
                    w=~np.isnan(bd[:, idx]),
                )

                self.bd_discr[1:-1, idx] = b_spline(self.ts_discr[1:-1])

            self.bd_discr[0] = np.nan
            self.bd_discr[-1] = np.nan

            for idx in range(tj.shape[1]):
                tj_spline = sp.interpolate.make_lsq_spline(
                    ts,
                    np.nan_to_num(tj[:, idx]),
                    np.array(4 * [ts[0]] + list(self.ts_discr) + 4 * [ts[-1]]),
                    w=~np.isnan(tj[:, idx]),
                )

                self.tj_discr[:, idx] = tj_spline(self.ts_discr)

            b_bg_x = np.interp(
                ts[1:-1][::5], self.ts_discr[2:-2], self.bd_discr[2:-2, 0]
            )
            b_bg_y = np.interp(
                ts[1:-1][::5], self.ts_discr[2:-2], self.bd_discr[2:-2, 1]
            )
            b_bg_z = np.interp(
                ts[1:-1][::5], self.ts_discr[2:-2], self.bd_discr[2:-2, 2]
            )

            assert not np.any(
                np.isnan(self.ts_discr)
            ), "ts_discr array cannot contain any NaN's"
            assert not np.any(
                np.isnan(self.bd_discr[1:-1])
            ), "bd_discr[1:-1] array cannot contain any NaN's"
            assert not np.any(
                np.isnan(self.tj_discr)
            ), "tj_discr array cannot contain any NaN's"

            self.noise_level = (
                np.nanstd(bd[1:-1, 0][::5] - b_bg_x)
                + np.nanstd(bd[1:-1, 1][::5] - b_bg_y)
                + np.nanstd(bd[1:-1, 2][::5] - b_bg_z)
            ) / 3
        else:
            self.ts_discr = np.linspace(t_start, t_end, self.discr, endpoint=True)

            self.bd_discr = np.zeros((self.discr, bd.shape[1]))
            self.tj_discr = np.zeros((self.discr, bd.shape[1]))

            for idx in range(bd.shape[1]):
                b_spline = sp.interpolate.make_lsq_spline(
                    ts,
                    np.nan_to_num(bd[:, idx]),
                    np.array(4 * [ts[0]] + list(self.ts_discr) + 4 * [ts[-1]]),
                    w=~np.isnan(bd[:, idx]),
                )

                self.bd_discr[:, idx] = b_spline(self.ts_discr)

            for idx in range(tj.shape[1]):
                tj_spline = sp.interpolate.make_lsq_spline(
                    ts,
                    np.nan_to_num(tj[:, idx]),
                    np.array(4 * [ts[0]] + list(self.ts_discr) + 4 * [ts[-1]]),
                    w=~np.isnan(tj[:, idx]),
                )

                self.tj_discr[:, idx] = tj_spline(self.ts_discr)

            b_bg_x = np.interp(ts[::5], self.ts_discr[1:-1], self.bd_discr[1:-1, 0])
            b_bg_y = np.interp(ts[::5], self.ts_discr[1:-1], self.bd_discr[1:-1, 1])
            b_bg_z = np.interp(ts[::5], self.ts_discr[1:-1], self.bd_discr[1:-1, 2])

            assert not np.any(
                np.isnan(self.ts_discr)
            ), "ts_discr array cannot contain any NaN's"
            assert not np.any(
                np.isnan(self.bd_discr)
            ), "bd_discr array cannot contain any NaN's"
            assert not np.any(
                np.isnan(self.tj_discr)
            ), "tj_discr array cannot contain any NaN's"

            self.noise_level = (
                np.nanstd(bd[:, 0][::5] - b_bg_x)
                + np.nanstd(bd[:, 1][::5] - b_bg_y)
                + np.nanstd(bd[:, 2][::5] - b_bg_z)
            ) / 3

        if bd.shape[1] == 1:
            self.scobs = ScObs(
                np.array([_ - ts[0] for _ in self.ts_discr]),
                self.tj_discr,
                self.bd_discr,
            )
        elif bd.shape[1] == 3:
            self.scobs = MagScObs(
                np.array([_ - ts[0] for _ in self.ts_discr]),
                self.tj_discr,
                self.bd_discr,
            )
        else:
            ValueError("ObserVec with {} dimensions is not supported", self.bd.shape[1])

        if "old" in kwargs:
            old = kwargs.get("old")

            if bd.shape[1] == 1:
                obser = Obser(old.scobs, len(old.pf))
            elif bd.shape[1] == 3:
                obser = MagObser(old.scobs, len(old.pf))

            self.pf = model.copy_pf(self.scobs, old.pf)
            self.pf.mag_simulate_with_errors(metric, obser)
        else:
            self.pf = model.new_pf(self.scobs, **kwargs)
            self.pf.mag_initialize(metric, threshold)

    def approximate_bayesian_computation(
        self,
        error_quantile=0.25,
        abort=1,
        metric="rmse",
        noise="gaussian",
    ):
        logger = lg.getLogger(__name__)

        statistics = []

        if not (0.05 <= error_quantile <= 0.95):
            raise ValueError(
                "error_quantile={:.3f} value must be within [0.05, 0.95]".format(
                    error_quantile
                )
            )

        if noise == "gaussian":
            noise_model = ObserVecNoise.gaussian(self.noise_level)
        else:
            raise NotImplemented("noise type {} is not implemented".format(noise))

        threshold = np.quantile(self.pf.errors(), error_quantile)

        while True:
            try:
                statistics.append(self.pf.mag_abc(metric, threshold, noise_model))
            except Exception as err:
                logger.warning(err)
                return statistics

            old_threshold = threshold
            errors = self.pf.errors()
            threshold = np.quantile(errors, error_quantile)

            logger.info(
                "abc_loop\n\teps: {:.3f} | {:.3f} - {:.3f} - {:.3f} (eps improv {:.2f}%)".format(
                    np.min(errors),
                    np.quantile(errors, 0.34),
                    np.quantile(errors, 0.5),
                    np.quantile(errors, 0.68),
                    100.0 * (old_threshold / threshold - 1.0),
                )
            )

            if 100.0 * (old_threshold / threshold - 1.0) < abort:
                break

        return statistics

    def differential_evolution(
        self,
        mutation_factor=0.1,
        recombination_factor=0.9,
        abort=1,
        metric="rmse",
        steps=50,
    ):
        """Use a differential evolution algorithm to reconstruct the observations.

        This process stops once the improvement w.r.t. the error metric is below the abort value (percentage value).

        Args:
            mutation_factor (float, optional): Differential evolution mutation factor. Defaults to 1.0.
            recombination_factor (float, optional): Differential evolution recombination factor. Defaults to 0.9.
            abort (float, optional): Abort threshold (percentage value). Defaults to 1.0.
            metric (str, optional): The error metric by name. Defaults to "rmse".
            steps (int, optional): Number of sub-iterations for the differential evolution algorithm.
                Small values may be problematic and cause early aborts. Defaults to 50.

        Raises:
            ValueError: If the abort criterion is outside the range (0, 10].
            ValueError: If the metric is unsupported.

        Returns:
            List[List[int]]: A list of lists containing the number of mutations per sub-iteration.
        """
        logger = lg.getLogger(__name__)

        mutations = []

        if not (0 < abort <= 10):
            raise ValueError(
                "abort={:.3f} criterion must be within (0, 10]".format(abort)
            )

        last_1sigma_error = np.quantile(self.pf.errors(), 0.34)

        while True:
            mutations.append(
                self.pf.mag_dev(metric, steps, mutation_factor, recombination_factor)
            )

            errors = self.pf.errors()

            logger.info(
                "dev_loop\n\teps: {:.3f} | {:.3f} - {:.3f} - {:.3f} (1-sigma improv {:.2f}%)".format(
                    np.min(errors),
                    np.quantile(errors, 0.34),
                    np.quantile(errors, 0.5),
                    np.quantile(errors, 0.68),
                    100.0 * (last_1sigma_error / np.quantile(errors, 0.34) - 1.0),
                )
            )

            if 100.0 * (last_1sigma_error / np.quantile(errors, 0.34) - 1.0) < abort:
                break
            else:
                last_1sigma_error = np.quantile(errors, 0.34)

        return mutations

    def sequential_importance_resampling(
        self,
        covariance,
        max_iterations=3,
        abort=None,
    ):
        logger = lg.getLogger(__name__)

        counter = 0
        statistics = []

        while counter < max_iterations:
            try:
                statistics.append(self.pf.mag_sir(covariance))
            except Exception as err:
                logger.warning(err)
                return statistics

            errors = self.pf.errors()

            logger.info(
                "sir_loop\n\tmvll: {:.1f} - {:.1f} (ess {:.1f})".format(
                    np.abs(np.min(errors)),
                    np.abs(np.max(errors)),
                    statistics[-1][0],
                )
            )

            if abort:
                if statistics[-1][0] < abort / 100 * len(self.pf):
                    break

            counter += 1

        return statistics
