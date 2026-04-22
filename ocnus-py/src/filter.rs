/// Implement python types for coronal rope ejection models.
macro_rules! impl_filter_obsvec {
    ($name: literal, $filter: literal, $model: ty, $obs: expr, $ndim: expr, $odim: expr, $params: expr, $csst: ty, $fmst: ty) => {
        paste! {
            #[pymethods]
            impl [<$name $filter Filter>] {
                /// Approximate Bayesian Computation iteration with Multivariate Normal Kernel.
                pub fn abc_mvnk<'py>(&mut self, py: Python<'py>, metric: String, threshold: Float, noise: &mut PyObsVecNoise) -> PyResult<Float> {
                    let error = select_error_metric!(metric, ObsVec<Float, $odim>);

                    py.detach(|| unroll_filter_errors!(self.0.abc_mvnk((&error, threshold), &$obs, &mut noise.0)))
                }

                /// Create a copy of the filter object.
                pub fn copy(&self) -> Self {
                    Self(self.0.clone())
                }

                /// Differential evolution step.
                pub fn dev<'py>(&mut self, py: Python<'py>, metric: String, mutation_factor: Float, recombination_factor: Float) -> PyResult<usize> {
                    let error = select_error_metric!(metric, ObsVec<Float, $odim>);

                    py.detach(|| unroll_model_errors!(self.0.dev(&error, &$obs,(mutation_factor, recombination_factor))))
                }

                /// Return the errors of the current observation ensemble.
                pub fn errors(&self) -> Vec<Float> {
                    self.0.errors().iter().cloned().collect()
                }

                /// Return the error quantile of the current observation ensemble.
                pub fn error_quantile(&self, value: Float) -> Float {
                    self.0.error_quantile(value).unwrap()
                }

                #[pyo3(signature = (metric="all".to_string(), threshold=1.0) )]
                /// Initialize the filter with a given error metric and threshold.
                pub fn initialize<'py>(&mut self, py: Python<'py>, metric: String, threshold: Float) -> PyResult<()> {
                    let error = select_error_metric!(metric, ObsVec<Float, $odim>);
                    let filter = |o1: &[ObsVec<Float, $odim>], o2: &[ObsVec<Float, $odim>]| (error(o1, o2) < threshold, error(o1, o2));

                    let prior = self.0.prior().clone();

                    py.detach(|| unroll_filter_errors!(self.0.initialize(&filter, &$obs, prior)))
                }

                /// Return the covariance matrix of the particle filter kernel.
                pub fn mvnk<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Float>> {
                    let mvnk: MultiNormalDensity<Float, Const<$params>, UDomain<Const<$params>>> =
                        MultiNormalDensity::from_view::<U1, Const<$params>>(
                            &self.0.particles().as_view(),
                            UDomain::new(Const::<$params>),
                            self.0.weights().map(|w| w.as_slice()),
                        )
                        .unwrap();

                    mvnk.get_covariance_matrix().to_pyarray(py)
                }

                /// Return the covariance matrix of the particle filter kernel.
                pub fn mvnk_ltm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<Float>> {
                    let mvnk: MultiNormalDensity<Float, Const<$params>, UDomain<Const<$params>>> =
                        MultiNormalDensity::from_view::<U1, Const<$params>>(
                            &self.0.particles().as_view(),
                            UDomain::new(Const::<$params>),
                            self.0.weights().map(|w| w.as_slice()),
                        )
                        .unwrap();

                    mvnk.get_lower_triangular_matrix().transpose().to_pyarray(py)
                }

                /// Return the likelihoods of the current observation ensemble given a covariance matrix.
                pub fn likelihoods<'py>(&self, py: Python<'py>, covariance: PyReadonlyArray2<Float>) -> PyResult<Vec<Float>> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(covariance)?;
                    let params = matrix.nrows();

                    let mvnpdf = MultiNormalDensity::from_matrix(matrix, DVector::zeros(params), UDomain::new(Dyn(params))).unwrap();

                    let llh = |o1: &[ObsVec<Float, $odim>], o2: &[ObsVec<Float, $odim>]| {
                        let mut value = ov_error(o1, o2, ObsVecMetric::Valid);

                        if value == 0.0 {
                            for i in 0..$odim {
                                let veca = DVector::from_iterator(params, o1.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let vecb = DVector::from_iterator(params, o2.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));

                                let delta = vecb - veca;

                                value -= mvnpdf.mahalanobis_distance_sq::<U1, Dyn>(&delta.as_view());
                            }
                        }

                        value
                    };

                    py.detach(|| {
                        Ok(self.0.errors_func(&llh))
                    })
                }

                /// Return the particles and their weights as numpy arrays.
                pub fn particles<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<Float>>, Option<Vec<Float>>) {
                    (
                        self.0.particles().transpose().to_pyarray(py),
                        self.0.weights().as_ref().map(|w| w.iter().cloned().collect::<Vec<Float>>())
                    )
                }

                #[pyo3(signature = (opt_obs = None))]
                /// Simulate observations and return the observation ensemble and errors.
                pub fn simulate(&mut self, opt_obs: Option<&PyObs>) -> PyResult<PyObsData> {
                    let obs_ensbl = match opt_obs {
                        Some(obs) => {
                            unroll_filter_errors!(self.0.simulate(Some(&obs.as_obs3()?.clone()), None, &$obs, &mut None::<&mut NullNoise<Float>>))

                        },
                        None => unroll_filter_errors!(self.0.simulate(None, None, &$obs, &mut None::<&mut NullNoise<Float>>)),

                    }?;

                    Ok(obs_ensbl.into())
                }

                #[pyo3(signature = (metric="rmse".to_string(), opt_obs = None, opt_ref_data = None))]
                /// Simulate observations and return the observation ensemble and errors.
                pub fn simulate_with_errors<'py>(&mut self, py: Python<'py>, metric: String, opt_obs: Option<&PyObs>, opt_ref_data: Option<&Bound<PyAny>>) -> PyResult<(PyObsData, Vec<Float>)> {
                    let error = select_error_metric!(metric, ObsVec<Float, $odim>);

                    let obs_ensbl = match (opt_obs, opt_ref_data) {
                        (Some(obs), Some(ref_data)) => {
                            let iter = py_any_iterator!(ref_data, [Float; $odim]).map(ObsVec::from);

                            unroll_filter_errors!(self.0.simulate(Some(&obs.as_obs3()?.clone()), Some(&DVector::from(Vec::from_iter(iter))), &$obs, &mut None::<&mut NullNoise<Float>>))

                        },
                        (None, None) => unroll_filter_errors!(self.0.simulate(None, None, &$obs, &mut None::<&mut NullNoise<Float>>)),
                        _ => Err(PyValueError::new_err("a new observation must be combined with a new reference data"))?,
                    }?;

                    py.detach(|| {
                        let errors = obs_ensbl.errors_func(&error);

                        Ok((obs_ensbl.into(), errors))
                    })
                }

                /// Sequential Importance Resampling iteration with Multivariate Normal Kernel.
                pub fn sir_mvnk<'py>(&mut self, py: Python<'py>, covariance: PyReadonlyArray2<Float>) -> PyResult<(Float, usize)> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(covariance)?;
                    let params = matrix.nrows();

                    let mvnpdf = MultiNormalDensity::from_matrix(matrix, DVector::zeros(params), UDomain::new(Dyn(params))).unwrap();

                    let llh = |o1: &[ObsVec<Float, $odim>], o2: &[ObsVec<Float, $odim>]| {
                        let mut value = ov_error(o1, o2, ObsVecMetric::Valid);

                        if value == 0.0 {
                            for i in 0..$odim {
                                let veca = DVector::from_iterator(params, o1.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));
                                let vecb = DVector::from_iterator(params, o2.iter().map(|ov| match ov[i].is_finite() {
                                    true => ov[i],
                                    false => 0.0,
                                }));
                                let delta = vecb - veca;

                                value -= mvnpdf.mahalanobis_distance_sq::<U1, Dyn>(&delta.as_view());
                            }
                        }

                        value
                    };

                    py.detach(|| {
                        unroll_filter_errors!(self.0.sir_mvnk(&$obs, &llh))
                    })
                }

                /// Return the size of the filter object ensemble.
                pub fn size(&self) -> usize {
                    self.0.len()
                }
            }
        }

        paste! {
            #[pymethods]
            impl [<$name Model>] {
                #[pyo3(signature = (obs, ref_data, size, initial_seed, **opt_kwargs))]
                /// Create a new filter object from a model.
                pub fn [<new_ $filter:lower _filter>]<'py>(&self, py: Python<'py>, obs: &PyObs, ref_data: &Bound<PyAny>, size: usize, initial_seed: u64, opt_kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<[<$name $filter Filter>]> {
                    let iter = py_any_iterator!(ref_data, [Float; $odim]).map(ObsVec::from);

                    let model_ensbl = ModelEnsbl::new(OMatrix::<Float, Const<$params>, Dyn>::zeros(size), None, None);

                    let obs_ensbl = py_unwrap!(ObsEnsbl::new(
                        obs.as_obs3()?.clone(),
                        size,
                        Some(DVector::from(Vec::from_iter(iter))),
                    ), "Observations must have same length as ref_data");

                    let mut fobj = py.detach(|| {
                        FilterObject::new(self.0.clone(), model_ensbl, obs_ensbl, initial_seed, None)
                    });

                    match opt_kwargs {
                        Some(kwargs) => {
                            match kwargs.get_item("exploration_factor")? {
                                Some(value) => fobj.settings.exploration_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("max_sample_attempts")? {
                                Some(value) => fobj.settings.max_sample_attempts = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("max_iterations")? {
                                Some(value) => fobj.settings.max_iterations = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("effective_particle_threshold_factor")? {
                                Some(value) => fobj.settings.effective_particle_threshold_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_ensemble_size_factor")? {
                                Some(value) => fobj.settings.simulation_ensemble_size_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_time_limit")? {
                                Some(value) => fobj.settings.simulation_time_limit = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_time_prediction")? {
                                Some(value) => fobj.settings.simulation_time_prediction = value.extract()?,
                                None =>(),
                            }
                        },
                        _ => ()
                    }

                    Ok([<$name $filter Filter>](fobj))

                }

                #[pyo3(signature = (obs, ref_data, input, initial_seed, **opt_kwargs))]
                /// Create a new filter object from a model and input ensemble.
                pub fn [<new_ $filter:lower _filter_with_particles>]<'py>(&self, py: Python<'py>, obs: &PyObs, ref_data: &Bound<PyAny>, input: PyReadonlyArray2<Float>, initial_seed: u64, opt_kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<[<$name $filter Filter>]> {
                    let iter = py_any_iterator!(ref_data, [Float; $odim]).map(ObsVec::from);
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    let model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);

                    let obs_ensbl = py_unwrap!(ObsEnsbl::new(
                        obs.as_obs3()?.clone(),
                        matrix.ncols(),
                        Some(DVector::from(Vec::from_iter(iter))),
                    ), "Observations must have same length as ref_data");

                    let mut fobj = py.detach(|| {
                        FilterObject::new(self.0.clone(), model_ensbl, obs_ensbl, initial_seed, None)
                    });


                    match opt_kwargs {
                        Some(kwargs) => {
                            match kwargs.get_item("exploration_factor")? {
                                Some(value) => fobj.settings.exploration_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("max_sample_attempts")? {
                                Some(value) => fobj.settings.max_sample_attempts = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("max_iterations")? {
                                Some(value) => fobj.settings.max_iterations = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("effective_particle_threshold_factor")? {
                                Some(value) => fobj.settings.effective_particle_threshold_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_ensemble_size_factor")? {
                                Some(value) => fobj.settings.simulation_ensemble_size_factor = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_time_limit")? {
                                Some(value) => fobj.settings.simulation_time_limit = value.extract()?,
                                None =>(),
                            }
                            match kwargs.get_item("simulation_time_prediction")? {
                                Some(value) => fobj.settings.simulation_time_prediction = value.extract()?,
                                None =>(),
                            }
                        },
                        _ => ()
                    };

                    Ok([<$name $filter Filter>](fobj))

                }
            }
        }
    };
}

pub(crate) use impl_filter_obsvec;
