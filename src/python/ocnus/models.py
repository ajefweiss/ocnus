# -*- coding: utf-8 -*-

import logging
from typing import Mapping

import numpy as np
import pyjson5 as json

# from .rotqs import generate_quaternions
from .util import rng_truncanted

lg = logging.getLogger(__name__)


class ModelContainer(object):
    dtype: np.float32

    model_params: dict
    model_params_array: np.ndarray

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
        params: str | Mapping,
        max_iterations: int = 50,
        overwrite_array: bool = False,
    ) -> None:
        if isinstance(params, str):
            # load from file
            self.model_params = json.decode_io(open(params, "rb", encoding="utf-8"))
        elif not hasattr(params, "__getitem__"):
            raise TypeError("params must be str or dict, not %s", type(params))
        else:
            self.model_params = params

        if hasattr(self, "model_params_array") and not overwrite_array:
            raise lg.warning(
                "model_params_array already exists, set overwrite_array to turn this warning off"
            )

        self.model_params_array = np.empty(
            (self.ensemble_size, len(self.model_params)), dtype=self.dtype
        )

        for key, param in self.model_params.items():
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

                self.model_params_array[:, index] = param["def_value"]
            elif distribution == "uniform":
                self.model_params_array[:, index] = (
                    np.random.rand(self.ensemble_size)
                    * (param["max_value"] - param["min_value"])
                    + param["min_value"]
                )
            else:
                # custom generator function, must accept size as parameter
                try:
                    identifiers = distribution.split(".")

                    assert identifiers[0] in ["np", "sp"]

                    func = globals()[identifiers[0]]

                    for name in identifiers[1:]:
                        func = getattr(func, name)

                    self.model_params_array[:, index] = rng_truncanted(
                        func,
                        param["max_value"],
                        param["min_value"],
                        self.ensemble_size,
                        max_iterations ** param["kwargs"],
                    )
                except Exception as e:
                    lg.error("failed to use distribution %s (%s)", distribution, e)
                    raise

        self._update_quaternions()

    def set_model_params(self, model_params_array: np.ndarray) -> None:
        if self.ensemble_size != len(model_params_array):
            raise ValueError(
                "array size does not match ensemble size (%i!=%i)",
                self.ensemble_size,
                len(model_params_array),
            )

        self.model_params_array = model_params_array
        self._update_quaternions()

    def _update_quaternions(self) -> None:
        if not hasattr(self, "quaternions"):
            self.quaterions = np.empty((self.ensemble_size, 4), dtype=self.dtype)

        # generate_quaternions(
        #     self.model_params_array, self.quaterions, indices=[0, 1, 2]
        # )

        # _rust.generate_quaternions()
