# -*- coding: utf-8 -*-

"""mfrfilter.py

Implements the ```MFRFilter``` class for an automatic fitting approach for flux rope models with magnetic field observations.
"""

import datetime as dt
import logging as lg
import numpy as np

from collections.abc import Iterable
from ocnus_py import Location3Series, Location3


class MFRFilter:
    def __init__(self, model, initial, timestamps, trajectory, reference_data, **kwargs):
        """Initialize an MFRFilter object.

        Args:
            model: A flux rope model object.
            timestamps: Nested list, corresponding to the observations timestamps.
            trajectory: Nested list, of spacecraft trajectory data. Must be of the form (N, L, 3).
            reference_data: Array of magnetic field reference data.
            **kwargs: Additional keyword arguments for configuration.

        Raises:
            TypeError: If the input data types are incorrect.
            ValueError: If the input data is invalid or inconsistent.
        """
        if not isinstance(timestamps, list):
            raise TypeError("timestamps must be a list")
        if not isinstance(trajectory, list):
            raise TypeError("trajectory must be a list")
        if not isinstance(reference_data, list):
            raise TypeError("reference_data must be a list")

        if not all(isinstance(_, Iterable) for _ in timestamps):
            raise TypeError("timestamps must be a list of iterables")
        if not all(isinstance(_, np.ndarray) for _ in trajectory):
            raise TypeError("trajectory must be a list of arrays")
        if not all(isinstance(_, np.ndarray) for _ in reference_data):
            raise TypeError("reference_data must be a list of ndarrays")

        for sub_timestamps in timestamps:
            # transfort any datetime objects to timestamps
            if all(isinstance(_, dt.datetime) for _ in sub_timestamps):
                for idx in range(len(sub_timestamps)):
                    sub_timestamps[idx] = sub_timestamps[idx].timestamp()
            elif all(isinstance(_, float) for _ in sub_timestamps):
                pass
            else:
                raise ValueError(
                    "each sublist in timestamps must be either a list of datetimes or timestamps (floats)"
                )

        conf = None

        for sub_timestamps, sub_trajectory in zip(timestamps, trajectory):
            # convert trajectory to numpy array
            sub_trajectory = np.array(sub_trajectory)

            new_obs = Location3Series(timestamps=sub_timestamps, opt_position=sub_trajectory)

            if conf is None:
                conf = new_obs
            else:
                conf = Location3Series.combine(conf, new_obs)

        self.conf = conf
        self.ref_data = np.hstack(reference_data)

        if kwargs.get("overwrite", False):
            raise NotImplementedError("overwrite functionality not yet implemented")
        else:
            self.filter = model.new_mag3_filter(Location3(initial), conf, self.ref_data.T, **kwargs)
            self.filter.initialize_mag3(
                metric=kwargs.get("metric", "nrmse"),
                threshold=kwargs.get("threshold", 1.0),
            )

    def dev(
        self,
        mutation_factor=0.1,
        recombination_factor=0.9,
        abort=1,
        metric="rmse",
        steps=50,
    ):
        """Perform differential evolution fitting.

        This process stops once the improvement w.r.t. the error metric is below the abort value (percentage value).

        Args:
            mutation_factor: Mutation factor for differential evolution.
            recombination_factor: Recombination factor for differential evolution.
            abort: Abort threshold (percentage value).
            metric: The error metric to use for fitting.
            steps: Number of sub-iterations for the differential evolution algorithm. Small values may be problematic and cause early aborts. Defaults to 50.

        Returns:
            A list of lists containing the number of mutations per sub-iteration.
        """
        logger = lg.getLogger(__name__)

        if not (0 < abort <= 10):
            raise ValueError(
                "abort={:.3f} criterion must be within (0, 10]".format(abort)
            )

        last_1sigma_error = np.quantile(self.filter.errors(), 0.1587)

        mutations = []

        while True:
            for _ in range(steps):
                mutations.append(
                    self.filter.dev_mag3(metric, mutation_factor, recombination_factor)
                )

            errors = self.filter.errors(), 

            logger.info(
                "dev_loop\n\teps: {:.3f} | {:.3f} - {:.3f} - {:.3f} (1-sigma improv {:.2f}%)".format(
                    np.min(errors),
                    np.quantile(errors, 0.1587),
                    np.quantile(errors, 0.5),
                    np.quantile(errors, 0.8413),
                    100.0 * (last_1sigma_error / np.quantile(errors, 0.1587) - 1.0),
                )
            )

            if 100.0 * (last_1sigma_error / np.quantile(errors, 0.1587) - 1.0) < abort:
                break
            else:
                last_1sigma_error = np.quantile(errors, 0.1587)

        return mutations
