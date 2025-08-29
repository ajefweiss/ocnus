use crate::{
    models::{unroll_model_errors, unroll_pf_errors},
    util::array_to_matrix,
    Float, PyCovMatrix, PyDiagObser, PyMagObser, PyMagScObs, PyObserVecNoise, PyUnivariate,
};
use nalgebra::{Const, DMatrix, Dyn, SVector, U1};
use numpy::{ndarray::Dim, PyArray2, PyReadonlyArray2, ToPyArray};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError, Obser},
    coords::Coordinates,
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    models,
    obsty::{
        observec_nchisq, observec_rmsep, observec_rmsetp, observec_valid, ICSCoordsBasis,
        InSituMagnetometer, NullNoise, ObserVec,
    },
    stats::MultivariateDensity,
};
use paste::paste;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyType},
};
use rayon::prelude::*;

/// Implement python types for single-observable vector models.
macro_rules! impl_vector_model {
    ($name: literal, $obsname: literal, $model: ty, $dim: expr, $csst: ty, $fmst: ty, $simulate: expr) => {
        paste! {
            #[allow(missing_docs)]
            #[derive(Clone)]
            #[pyclass]
            pub struct [<$name Ensbl>](pub ModelEnsbl<Float, $model<Float, MultivariateDensity<Float, $dim>>, $dim>);
        }

        paste! {
            #[pymethods]
            impl [<$name Ensbl>] {
                #[allow(missing_docs)]
                pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
                    let column = self.0.ptpdf.get_particle(key);

                    let matrix = DMatrix::from_iterator(
                        $dim,
                        1,
                        column.iter().copied(),
                    );

                    matrix.transpose().to_pyarray(py)
                }

                #[allow(missing_docs)]
                #[classmethod]
                pub fn from_particles(_cls: &Bound<PyType>, array: PyReadonlyArray2<Float>) -> PyResult<Self> {
                    Ok(Self(ModelEnsbl::from_particles(array_to_matrix::<Dim<[usize; 2]>, Dyn, Const<$dim>, U1, Dyn>(array)?, None, None)))
                }

                #[allow(missing_docs)]
                #[new]
                pub fn new(size: usize) -> Self {
                    Self(ModelEnsbl::new(size, None))
                }

                #[allow(missing_docs)]
                pub fn set_particle(&mut self, index: usize, values: &Bound<PyAny>) -> PyResult<()> {
                    let iter = match values.try_iter() {
                        Ok(value) => value.map(|py_obj| py_obj.unwrap().extract::<Float>().unwrap()),
                        Err(..) => return Err(PyValueError::new_err("values argument must be iterable")),
                    };

                    let vector = SVector::<Float, $dim>::from_iterator(iter);

                    self.0.ptpdf.set_particle(index, &vector.as_view());

                    Ok(())
                }
            }
        }

        paste! {
            #[allow(missing_docs)]
            #[derive(Clone)]
            #[pyclass]
            pub struct [<$name ParticleFilter>] {
                pub pf: ParticleFilter<Float, $model<Float, MultivariateDensity<Float, $dim>>, $dim, ObserVec<Float, 3>>,
                pub settings: ParticleFilterSettings<Float>,
            }
        }

        paste! {
            #[pymethods]
            impl [<$name ParticleFilter>] {
                #[allow(missing_docs)]
                pub fn __len__(&self) -> usize {
                    self.pf.ensbl.len()
                }

                #[allow(missing_docs)]
                pub fn copy(&self) -> Self {
                    self.clone()
                }

                #[allow(missing_docs)]
                pub fn errors(&self) -> Vec<Float> {
                    self.pf.errors.clone()
                }

                #[allow(missing_docs)]
                pub fn error_quantile(&self, value: Float) -> Float {
                    self.pf.error_quantile(value).unwrap()
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _abc>]<'py>(&mut self, py: Python<'py>, metric: String, threshold: Float, noise: &mut PyObserVecNoise) -> PyResult<(Float, Float)> {
                    match metric.to_lowercase().as_str() {
                        "nchisqr" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_nchisq(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_abc(&self.settings, &mut noise.0, &$simulate, (&error, threshold)))
                            })
                        },
                        "rmse" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_rmsep(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_abc(&self.settings, &mut noise.0, &$simulate, (&error, threshold)))
                            })
                        },
                        "rmset" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_rmsetp(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_abc(&self.settings, &mut noise.0, &$simulate, (&error, threshold)))
                            })
                        },
                        _ => Err(PyValueError::new_err("unsupported metric"))
                    }

                }

                #[allow(missing_docs)]
                pub fn [<$obsname _dev>]<'py>(&mut self, py: Python<'py>, metric: String, max_iterations: usize, mutation_factor: Float, recombination_factor: Float) -> PyResult<Vec<usize>> {
                    match metric.to_lowercase().as_str() {
                        "nchisqr" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_nchisq(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_model_errors!(self.pf.pf_dev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                            })
                        }
                        "rmse" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_rmsep(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_model_errors!(self.pf.pf_dev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                            })
                        },
                        "rmset" => {
                            let error = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                observec_rmsetp(o1, o2)
                            };

                            py.allow_threads(|| {
                                unroll_model_errors!(self.pf.pf_dev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                            })
                        },
                        _ => Err(PyValueError::new_err("unsupported metric"))
                    }
                }

                #[allow(missing_docs)]
                #[pyo3(signature = (metric="all".to_string(), threshold=1.0) )]
                pub fn [<$obsname _initialize>]<'py>(&mut self, py: Python<'py>, metric: String, threshold: Float) -> PyResult<()> {
                    match metric.to_lowercase().as_str() {
                        "all" | "none" => {
                            let filter = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                let value = observec_valid(o1, o2);

                                (value, Float::INFINITY, )
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                            })
                        }
                        "nchisqr" => {
                            let filter = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                let value = observec_nchisq(o1, o2);

                                (value < threshold, value, )
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                            })
                        }
                        "rmse" => {
                            let filter = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                let value = observec_rmsep(o1, o2);

                                (value < threshold, value, )
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                            })
                        },
                        "rmset" => {
                            let filter = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                                let value = observec_rmsetp(o1, o2);

                                (value < threshold, value, )
                            };

                            py.allow_threads(|| {
                                unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                            })
                        },
                        _ => Err(PyValueError::new_err("unsupported metric"))
                    }
                }

                #[allow(missing_docs)]
                #[pyo3(signature = (metric="rmse".to_string(), opt_obser=None))]
                pub fn [<$obsname _simulate_with_errors>]<'py>(&mut self, py: Python<'py>, metric: String, opt_obser: Option<&mut PyMagObser>) -> PyResult<()> {
                    self.pf.model.initialize_states_ensbl(&mut self.pf.ensbl).unwrap();

                    match metric.to_lowercase().as_str() {
                        "nchisqr" => {
                            py.allow_threads(|| {
                                let obser = match opt_obser {
                                    Some(value) => &mut value.0,
                                    None => &mut self.pf.obser
                                };

                                self.pf.model.simulate_ensbl(&mut self.pf.ensbl, obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                                self.pf.errors = obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_nchisq(obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();
                            });

                            Ok(())
                        }
                        "rmse" => {
                            py.allow_threads(|| {
                                let obser = match opt_obser {
                                    Some(value) => &mut value.0,
                                    None => &mut self.pf.obser
                                };

                                self.pf.model.simulate_ensbl(&mut self.pf.ensbl, obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                                self.pf.errors = obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_rmsep(obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();
                            });

                            Ok(())
                        },
                        "rmset" => {
                            py.allow_threads(|| {
                                let obser = match opt_obser {
                                    Some(value) => &mut value.0,
                                    None => &mut self.pf.obser
                                };

                                self.pf.model.simulate_ensbl(&mut self.pf.ensbl, obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                                self.pf.errors = obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_rmsetp(obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();
                            });

                            Ok(())
                        },
                        _ => Err(PyValueError::new_err("unsupported metric"))
                    }
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _sir>]<'py>(&mut self, py: Python<'py>, covm: &PyCovMatrix) -> PyResult<(Float, usize, Float)> {
                    let llh = |o1: &[ObserVec<Float, 3>], o2: &[ObserVec<Float, 3>]| {
                        covm.0.observec_log_likelihood(o1, o2)
                    };

                    py.allow_threads(|| {
                        unroll_pf_errors!(self.pf.pf_sir(&self.settings, &$simulate, &llh))
                    })
                }

                #[allow(missing_docs)]
                pub fn particles<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<Float>>, Vec<Float>) {
                    (
                        self.pf.ensbl.ptpdf.particles().transpose().to_pyarray(py),
                        self.pf.ensbl.ptpdf.weights().iter().cloned().collect::<Vec<Float>>()
                    )
                }
            }
        }

        paste! {
            #[allow(missing_docs)]
            #[pyclass]
            pub struct [<$name Model>](pub $model<Float, MultivariateDensity<Float, $dim>>);}

            paste! {
            #[pymethods]
            impl [<$name Model>] {
                #[allow(missing_docs)]
                pub fn copy_pf(&self, scobs: &PyMagScObs, old_pf: &[<$name ParticleFilter>]) -> PyResult<[<$name ParticleFilter>]> {
                    Ok(
                        [<$name ParticleFilter>] {
                            pf: ParticleFilter::from_particles(scobs.0.clone(), self.0.clone(), old_pf.pf.ensbl.ptpdf.particles().clone(), Some(old_pf.pf.ensbl.ptpdf.weights().clone()), old_pf.pf.rseed),
                            settings: old_pf.settings.clone()
                        }
                    )
                }

                #[allow(missing_docs)]
                pub fn diagnostics<'py>(&self, py: Python<'py>, scobs: &PyMagScObs, ensbl: &mut [<$name Ensbl>]) -> PyDiagObser {
                    py.allow_threads(|| {
                        self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                        let mut obser = Obser::<Float, ICSCoordsBasis<Float>>::new(scobs.0.as_scobs(), ensbl.0.len());

                        self.0.simulate_ics_basis_ensbl(&mut ensbl.0, &mut obser).unwrap();

                        PyDiagObser(obser)
                    })
                }

                #[allow(missing_docs)]
                #[new]
                pub fn new(priors: Vec<PyUnivariate>) -> PyResult<Self> {
                    let names = $model::<Float, MultivariateDensity<Float, $dim>>::PARAMS;

                    if priors.len() != $dim {
                        Err(PyValueError::new_err("invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.0.0 == *next.1)
                        }) {
                            let mvpdf = MultivariateDensity::new(priors.iter().map(|uvpdf| &uvpdf.0 .1));

                            Ok(Self($model::<Float, MultivariateDensity<Float, $dim>>::new(mvpdf)))
                        } else {
                            Err(PyValueError::new_err("invalid parameter names"))
                        }
                    }
                }

                #[allow(missing_docs)]
                #[pyo3(signature = (scobs, size, initial_seed, **opt_kwargs))]
                pub fn new_pf(&self, scobs: &PyMagScObs, size: usize, initial_seed: u64, opt_kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<[<$name ParticleFilter>]> {
                    Ok(match opt_kwargs {
                        Some(kwargs) => {
                            [<$name ParticleFilter>] {
                                pf: ParticleFilter::new(scobs.0.clone(), self.0.clone(), size, initial_seed),
                                settings: ParticleFilterSettings {
                                    expl_factor: match kwargs.get_item("expl_factor")? {
                                        Some(value) => value.extract()?,
                                        None => 2.0
                                    },
                                    max_attempts: match kwargs.get_item("max_attempts")? {
                                        Some(value) => value.extract()?,
                                        None => 250
                                    },
                                    max_iterations: match kwargs.get_item("max_iterations")? {
                                        Some(value) => value.extract()?,
                                        None => 10
                                    },
                                    eff_particle_threshold_factor: match kwargs.get_item("eff_particle_threshold_factor")? {
                                        Some(value) => value.extract()?,
                                        None => 0.05
                                    },
                                    simulation_ensemble_size_factor: match kwargs.get_item("simulation_ensemble_size_factor")? {
                                        Some(value) => value.extract()?,
                                        None => 4
                                    },
                                    simulation_time_limit: match kwargs.get_item("simulation_time_limit")? {
                                        Some(value) => value.extract()?,
                                        None => 10.0
                                    },
                                    simulation_time_prediction: match kwargs.get_item("simulation_time_prediction")? {
                                        Some(value) => value.extract()?,
                                        None => true
                                    },
                                }
                            }
                        },
                        None => {
                            [<$name ParticleFilter>] {
                                pf: ParticleFilter::new(scobs.0.clone(), self.0.clone(), size, initial_seed),
                                settings: ParticleFilterSettings {
                                    expl_factor: 2.0,
                                    max_attempts: 250,
                                    max_iterations: 10,
                                    eff_particle_threshold_factor: 0.05,
                                    simulation_ensemble_size_factor: 4,
                                    simulation_time_limit: 10.0,
                                    simulation_time_prediction: true,
                                },
                            }
                        }
                    })
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _simulate>](&self, ensbl: &mut [<$name Ensbl>], obser: &mut PyMagObser)  {
                    self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                    self.0.simulate_ensbl(&mut ensbl.0, &mut obser.0, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _fisher>]<'py>(&self, py: Python<'py>, scobs: &PyMagScObs, values: &Bound<PyAny>, covm: &PyCovMatrix) -> PyResult<Bound<'py, PyArray2<Float>>> {
                    let iter = match values.try_iter() {
                        Ok(value) => value.map(|py_obj| py_obj.unwrap().extract::<Float>().unwrap()),
                        Err(..) => return Err(PyValueError::new_err("values argument must be iterable")),
                    };

                    let vector = SVector::<Float, $dim>::from_iterator(iter);

                    let matrix = self.0.[<fisher_ $obsname>](&scobs.0, &vector.as_view::<Const<$dim>, U1, U1, Const<$dim>>(), &covm.0).unwrap();

                    Ok(matrix.transpose().to_pyarray(py))
                }
            }
        }
    };
}

impl_vector_model!(
    "NC16",
    "mag",
    models::NC16Model,
    9,
    XCState<Float>,
    (),
    models::NC16Model::observe_mag3
);

impl_vector_model!(
    "ECH",
    "mag",
    models::ECHModel,
    12,
    XCState<Float>,
    (),
    models::ECHModel::observe_mag3
);

impl_vector_model!(
    "CORE",
    "mag",
    models::COREModel,
    11,
    coordinate::TTState<Float>,
    models::COREState<Float>,
    models::COREModel::observe_mag3
);
