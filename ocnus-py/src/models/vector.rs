use crate::{
    models::{unroll_model_errors, unroll_pf_errors},
    Float, PyCovMatrix, PyDiagObser, PyMagObser, PyMagScObs, PyObserVecNoise, PyUnivariate,
};
use nalgebra::{Const, DMatrix, Dyn, SVector, U1};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError, Obser},
    coords::Coordinates,
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    models,
    obsty::{observec_rmsep, observec_trmsep, ICSCoordsBasis, MeasInSituMag, NullNoise, ObserVec},
    stats::MultivariateDensity,
};
use paste::paste;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use rayon::prelude::*;

/// Implement python types for single-observable vector models.
macro_rules! impl_vector_model {
    ($name: literal, $obsname: literal, $model: ty, $dim: expr, $vdim: expr, $csst: ty, $fmst: ty, $simulate: expr) => {
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
                #[new]
                pub fn new(size: usize) -> Self {
                    Self(ModelEnsbl::new(size, None))
                }

                #[allow(missing_docs)]
                pub fn set_particle(&mut self, index: usize, values: &Bound<PyAny>) {
                    let iter = values.try_iter().expect("values must be iterable").map(|py_obj| py_obj.unwrap().extract::<Float>().unwrap());
                    let vector = SVector::<Float, $dim>::from_iterator(iter);

                    self.0.ptpdf.set_particle(index, &vector.as_view());
                }
            }
        }

        paste! {
            #[allow(missing_docs)]
            #[derive(Clone)]
            #[pyclass]
            pub struct [<$name Obser>](pub Obser<Float, ObserVec<Float, $vdim>>);
        }

        paste! {
            #[pymethods]
            impl [<$name Obser>] {
                #[allow(missing_docs)]
                #[new]
                pub fn new(scobs: &PyMagScObs, size: usize) -> PyResult<Self> {
                    Ok(Self(Obser::new(scobs.0.clone(), size)))
                }

                #[allow(missing_docs)]
                pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
                    let column = self.0.get_output(key);

                    let matrix = DMatrix::from_iterator(
                        3,
                        self.0.len(),
                        column.iter().flat_map(|icscb| icscb.iter().copied()),
                    );

                    matrix.transpose().to_pyarray(py)
                }
            }
        }

        paste! {
            #[allow(missing_docs)]
            #[derive(Clone)]
            #[pyclass]
            pub struct [<$name ParticleFilter>] {
                pub pf: ParticleFilter<Float, $model<Float, MultivariateDensity<Float, $dim>>, $dim, ObserVec<Float, $vdim>>,
                pub settings: ParticleFilterSettings<Float>,
            }
        }

        paste! {
            #[pymethods]
            impl [<$name ParticleFilter>] {
                #[allow(missing_docs)]
                pub fn errors(&self) -> Vec<Float> {
                    self.pf.errors.clone()
                }

                #[allow(missing_docs)]
                pub fn error_quantile(&self, value: Float) -> Float {
                    self.pf.error_quantile(value).unwrap()
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _abc_rmse>](&mut self, noise: &mut PyObserVecNoise) -> PyResult<(Vec<Float>, Vec<Float>)> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_rmsep(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_abc_loop(&self.settings, &mut noise.0, &$simulate, &error))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _abc_iter_rmse>](&mut self, noise: &mut PyObserVecNoise, threshold: Float) -> PyResult<(Float, Float)> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_rmsep(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_abc_iter(&self.settings, &mut noise.0, &$simulate, (&error, threshold)))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _dev_rmse>](&mut self, max_iterations: usize, mutation_factor: Float, recombination_factor: Float) -> PyResult<Vec<usize>> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_rmsep(o1, o2)
                    };

                    unroll_model_errors!(self.pf.diff_ev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _dev_trmse>](&mut self, max_iterations: usize, mutation_factor: Float, recombination_factor: Float) -> PyResult<Vec<usize>> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_trmsep(o1, o2)
                    };

                    unroll_model_errors!(self.pf.diff_ev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _initialize_rmse>](&mut self, rmse_percentage: Float) -> PyResult<()> {
                    let filter = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        let value = observec_rmsep(o1, o2);

                        (value < rmse_percentage, value, )
                    };

                    unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _sir_mvllh>](&mut self, covm: &PyCovMatrix) -> PyResult<(Vec<usize>, Vec<Float>)> {
                    let llh = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        covm.0.observec_log_likelihood(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_sir_loop(&self.settings, &$simulate, &llh))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _sir_iter_mvllh>](&mut self, covm: &PyCovMatrix) -> PyResult<(usize, Float)> {
                    let llh = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        covm.0.observec_log_likelihood(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_sir_iter(&self.settings, &$simulate, &llh))
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
                pub fn diagnostics(&self, scobs: &PyMagScObs, ensbl: &mut [<$name Ensbl>]) -> PyDiagObser {
                    self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                    let mut obser = Obser::<Float, ICSCoordsBasis<Float>>::new(scobs.0.as_scobs(), ensbl.0.len());

                    self.0.simulate_ics_basis_ensbl(&mut ensbl.0, &mut obser).unwrap();

                    PyDiagObser(obser)
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _simulate_rmse>](&self, array: PyReadonlyArray2<Float>, scobs: &PyMagScObs) -> (PyMagObser, Vec<Float>)  {
                    let mut ensbl = ModelEnsbl::<Float, $model::<Float, MultivariateDensity<Float, $dim>>, $dim>::from_particles(
                        array
                        .try_as_matrix::<Dyn, Const<$dim>, U1, Dyn>().expect("failed to convert numpy array to matrix").transpose().into_owned()
                    );

                    self.0.initialize_states_ensbl(&mut ensbl).unwrap();

                    let mut obser = Obser::new(scobs.0.clone(), ensbl.len());

                    self.0.simulate_ensbl(&mut ensbl, &mut obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                    let errors = obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_rmsep(obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();

                    (PyMagObser(obser), errors)
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
                                    error_quantile: match kwargs.get_item("error_quantile")? {
                                        Some(value) => value.extract()?,
                                        None => 0.25
                                    },
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
                                        None => 0.1
                                    },
                                    simulation_ensemble_size_factor: match kwargs.get_item("simulation_ensemble_size_factor")? {
                                        Some(value) => value.extract()?,
                                        None => 4
                                    },
                                    simulation_time_limit: match kwargs.get_item("simulation_time_limit")? {
                                        Some(value) => value.extract()?,
                                        None => 10.0
                                    },
                                }
                            }
                        },
                        None => {
                            [<$name ParticleFilter>] {
                                pf: ParticleFilter::new(scobs.0.clone(), self.0.clone(), size, initial_seed),
                                settings: ParticleFilterSettings {
                                    error_quantile: 0.25,
                                    expl_factor: 2.0,
                                    max_attempts: 250,
                                    max_iterations: 10,
                                    eff_particle_threshold_factor: 0.1,
                                    simulation_ensemble_size_factor: 4,
                                    simulation_time_limit: 10.0
                                },
                            }
                        }
                })
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _simulate>](&self, ensbl: &mut [<$name Ensbl>], obser: &mut [<$name Obser>])  {
                    self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                    self.0.simulate_ensbl(&mut ensbl.0, &mut obser.0, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _fisher>]<'py>(&self, py: Python<'py>, scobs: &PyMagScObs, values: &Bound<PyAny>, covm: &PyCovMatrix) -> Bound<'py, PyArray2<Float>> {
                    let iter = values.try_iter().expect("values must be iterable").map(|py_obj| py_obj.unwrap().extract::<Float>().unwrap());
                    let vector = SVector::<Float, $dim>::from_iterator(iter);

                    let matrix = self.0.[<fisher_ $obsname>](&scobs.0, &vector.as_view::<Const<$dim>, U1, U1, Const<$dim>>(), &covm.0).unwrap();

                    matrix.transpose().to_pyarray(py)
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
    3,
    XCState<Float>,
    (),
    models::NC16Model::observe_mag3
);

impl_vector_model!(
    "ECH",
    "mag",
    models::ECHModel,
    12,
    3,
    XCState<Float>,
    (),
    models::ECHModel::observe_mag3
);

impl_vector_model!(
    "CORE",
    "mag",
    models::COREModel,
    11,
    3,
    coordinate::TTState<Float>,
    models::COREState<Float>,
    models::COREModel::observe_mag3
);
