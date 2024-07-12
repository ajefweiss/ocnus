# -*- coding: utf-8 -*-

import logging
from typing import Iterable, Mapping

import numpy as np
import pyjson5 as json
import scipy as sp  # noqa: F401

# from .rotqs import generate_quaternions


lg = logging.getLogger(__name__)


class ModelContainer(object):
    """Generic model container to handle a variety of different magnetic flux rope models.

    Implcitly assumes that any model lives within its own Cartesian coordinate system,
    with a propagation direction along the x-axis. Any model must implement a rot_z/y/x parameter
    (which are the first three given parameters) with which one can rotate the geometry back
    into a heliospheric coordinate system.
    """

    dtype: type

    parameters: dict

    param_larr: Iterable[np.ndarray]  # list of arrays holding the model parameters
    state_larr: Iterable[np.ndarray]  # list of arrays holding the current model state

    quaterions: np.ndarray

    ensemble_size: int

    random_seed: int

    def __init__(
        self,
        ensemble_size: int = 1,
        random_seed: int = 42,
        dtype: type = np.float32,
    ) -> None:
        assert isinstance(dtype, type)
        self.dtype = dtype

        assert isinstance(ensemble_size, int)
        self.ensemble_size = ensemble_size

        assert isinstance(random_seed, int)
        self.random_seed = random_seed

    def generate_model_params(
        self,
        parameters: str | Mapping,
        max_iterations: int = 50,
        overwrite_array: bool = False,
    ) -> None:
        """Generate initial model parameters.

        Args:
            parameters (str | Mapping): path to json file or equivalent dict object.
            max_iterations (int, optional): maximum number of iterations allowed for the truncated random number
                                            generator. Defaults to 50.
            overwrite_array (bool, optional): allow overwriting of existing model parameters (warning only).
                                              Defaults to False.

        Raises:
            TypeError: invalid parameters argument
            ValueError: invalid parameters range (common mistake is max_value < min_value)
        """
        from .util import rng_truncanted

        if isinstance(parameters, str):
            # load from file
            self.parameters = json.decode_io(open(parameters, "rb", encoding="utf-8"))
        elif not hasattr(parameters, "__getitem__"):
            raise TypeError("parameters must be str or dict, not %s", type(parameters))
        else:
            self.parameters = parameters

        if hasattr(self, "param_larr") and not overwrite_array:
            lg.warning(
                "param_larr already exists, set overwrite_array to turn this warning off"
            )

        self.param_larr = np.empty(
            (self.ensemble_size, len(self.parameters)), dtype=self.dtype
        )

        for key, param in self.parameters.items():
            index = param["param_index"]
            distribution = param["distribution"]

            if param["max_value"] <= param["min_value"]:
                raise ValueError(
                    "invalid parameter range for %s, max_value < min_value",
                    param.get("label", index),
                )

            if distribution == "fixed":
                if not param["max_value"] > param["def_value"] > param["min_value"]:
                    raise ValueError(
                        "invalid parameter range for %s, def_value outside of valid range",
                        param.get("label", index),
                    )

                self.param_larr[:, index] = param["def_value"]
            elif distribution == "uniform":
                self.param_larr[:, index] = (
                    np.random.rand(self.ensemble_size)
                    * (param["max_value"] - param["min_value"])
                    + param["min_value"]
                )
            else:
                # custom generator function, must accept size as parameter
                try:
                    identifiers = distribution.split(".")

                    # only allow for numpy or scipy functions
                    assert identifiers[0] in ["np", "sp"]

                    func = globals()[identifiers[0]]

                    for name in identifiers[1:]:
                        func = getattr(func, name)

                    self.param_larr[:, index] = rng_truncanted(
                        func,
                        param["max_value"],
                        param["min_value"],
                        self.ensemble_size,
                        max_iterations ** param["kwargs"],
                    )
                except RuntimeError as e:
                    lg.error("failed to use distribution %s (%s)", distribution, e)
                    raise

        # generate quaternions for new model parameters
        self._update_quaternions()

    def set_model_params(self, param_larr: np.ndarray) -> None:
        if self.ensemble_size != len(param_larr):
            raise ValueError(
                "array size does not match ensemble size (%i!=%i)",
                self.ensemble_size,
                len(param_larr),
            )

        self.param_larr = param_larr
        self._update_quaternions()

    def _update_quaternions(self) -> None:
        if not hasattr(self, "quaternions"):
            self.quaterions = np.empty((self.ensemble_size, 4), dtype=self.dtype)

        # generate_quaternions(
        #     self.param_larr, self.quaterions, indices=[0, 1, 2]
        # )

        # _rust.generate_quaternions()
