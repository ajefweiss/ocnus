use crate::{
    models::{unroll_model_errors, unroll_pf_errors},
    util::array_to_matrix,
    Float, PyCovMatrix, PyDiagObser, PyObser, PyObserVecNoise, PyScObs, PyUnivariate,
};
use nalgebra::{DMatrix, Dyn, SVector};
use numpy::{
    ndarray::Dim, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray,
};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError, Obser},
    coords::Coordinates,
    methods::filters::{ParticleFilter, ParticleFilterError, ParticleFilterSettings},
    models::{self, WSAHUXModel, WSAInputData},
    obsty::{observec_rmsep, ICSCoordsBasis, InSituPlasmaBulkVelocity, NullNoise, ObserVec},
    stats::MultivariateDensity,
};
use paste::paste;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyString, PyType},
};
use rayon::prelude::*;

#[allow(missing_docs)]
#[pyclass(name = "WSAInputData")]
pub struct PyWSAInputData(pub WSAInputData<Float>);

#[pymethods]
impl PyWSAInputData {
    #[allow(missing_docs)]
    #[classmethod]
    pub fn load(_cls: &Bound<PyType>, path: &Bound<PyString>) -> PyResult<Self> {
        match serde_json5::from_str::<WSAInputData<Float>>(&std::fs::read_to_string(
            path.extract::<String>().unwrap(),
        )?) {
            Ok(value) => Ok(Self(value)),
            Err(_) => Err(PyValueError::new_err("failed to load from file")),
        }
    }

    #[allow(missing_docs)]
    #[new]
    #[pyo3(signature = (dmap, efs, opt_lon_1d = None, opt_lat_1d = None, opt_lat_indices = None))]
    pub fn new(
        dmap: PyReadonlyArray2<Float>,
        efs: PyReadonlyArray2<Float>,
        opt_lon_1d: Option<PyReadonlyArray1<Float>>,
        opt_lat_1d: Option<PyReadonlyArray1<Float>>,
        opt_lat_indices: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let lon_count = dmap.shape()[0];
        let lat_count = dmap.shape()[1];

        let lon_1d = match opt_lon_1d {
            Some(lon_1d) => array_to_matrix::<Dim<[usize; 1]>, Dyn, Dyn, Dyn, Dyn>(lon_1d)?,
            None => DMatrix::from_iterator(
                lon_count,
                1,
                (0..lon_count)
                    .map(|idx| idx as Float / lon_count as Float * std::f64::consts::TAU as Float),
            ),
        };

        let lat_1d = match opt_lat_1d {
            Some(lat_1d) => array_to_matrix::<Dim<[usize; 1]>, Dyn, Dyn, Dyn, Dyn>(lat_1d)?,
            None => DMatrix::from_iterator(
                lat_count,
                1,
                (0..lat_count).map(|idx| {
                    (-0.5 + idx as Float / lat_count as Float) * std::f64::consts::PI as Float
                }),
            ),
        };

        Ok(Self(WSAInputData {
            lon_1d,
            lat_1d,
            dmap: array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(dmap)?,
            efs: array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(efs)?,
            lat_indices: opt_lat_indices.unwrap_or_default(),
        }))
    }
}

/// Implement python types for single-observable vector models.
macro_rules! impl_wsahux_model {
    ($name: literal, $obsname: literal, $model: ty, $rcount: expr, $dim: expr, $vdim: expr, $csst: ty, $fmst: ty, $simulate: expr) => {
        paste! {
            #[allow(missing_docs)]
            #[pyclass]
            pub struct [<$name $rcount Ensbl>](pub ModelEnsbl<Float, WSAHUXModel<Float, $rcount, MultivariateDensity<Float, $dim>>, $dim>);
        }

        paste! {
            #[pymethods]
            impl [<$name $rcount Ensbl>] {
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
            pub struct [<$name $rcount Obser>](pub Obser<Float, ObserVec<Float, $vdim>>);
        }

        paste! {
            #[pymethods]
            impl [<$name $rcount Obser>] {
                #[allow(missing_docs)]
                #[new]
                pub fn new(scobs: &PyScObs, size: usize) -> PyResult<Self> {
                    Ok(Self(Obser::new(scobs.0.clone(), size)))
                }

                #[allow(missing_docs)]
                pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
                    let column = self.0.get_output(key);

                    let matrix = DMatrix::from_iterator(
                        1,
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
            pub struct [<$name $rcount ParticleFilter>] {
                pub pf: ParticleFilter<Float, $model<Float, $rcount, MultivariateDensity<Float, $dim>>, $dim, ObserVec<Float, $vdim>>,
                pub settings: ParticleFilterSettings<Float>,
            }
        }

        paste! {
            #[pymethods]
            impl [<$name $rcount ParticleFilter>] {
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
                pub fn extract_slice<'py>(&self, py: Python<'py>, index: usize, latitude: Float) -> Bound<'py, PyArray2<Float>> {
                    let lat_index = self.pf.model.latitude_index(latitude.to_radians());

                    self.pf.ensbl.fm_states[index].wsahux[lat_index].0.to_pyarray(py)
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _abc_rmse>]<'py>(&mut self, py: Python<'py>, noise: &mut PyObserVecNoise, threshold: Float) -> PyResult<(Float, Float)> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_rmsep(o1, o2)
                    };

                    py.allow_threads(|| {
                        unroll_pf_errors!(self.pf.pf_abc(&self.settings, &mut noise.0, &$simulate, (&error, threshold)))
                    })
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _dev_rmse>]<'py>(&mut self, py: Python<'py>, max_iterations: usize, mutation_factor: Float, recombination_factor: Float) -> PyResult<Vec<usize>> {
                    let error = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        observec_rmsep(o1, o2)
                    };

                    py.allow_threads(|| {
                        unroll_model_errors!(self.pf.pf_dev_loop(max_iterations, (mutation_factor, recombination_factor), &$simulate, &error))
                    })
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _initialize_rmse>]<'py>(&mut self, py: Python<'py>, rmse_percentage: Float) -> PyResult<()> {
                    let filter = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        let value = observec_rmsep(o1, o2);

                        (value < rmse_percentage, value, )
                    };

                    py.allow_threads(|| {
                        unroll_pf_errors!(self.pf.pf_initialize_ensbl(&filter, &$simulate, &self.settings))
                    })
                }

                #[allow(missing_docs)]
                #[pyo3(signature = (opt_scobs = None))]
                pub fn [<$obsname _simulate_rmse>]<'py>(&mut self, py: Python<'py>, opt_scobs: Option<&PyScObs>) -> PyObser  {
                    py.allow_threads(|| {
                        self.pf.model.initialize_states_ensbl(&mut self.pf.ensbl).unwrap();

                        match opt_scobs {
                            Some(scobs) => {
                                let mut obser = Obser::new(scobs.0.clone(), self.pf.ensbl.len());

                                self.pf.model.simulate_ensbl(&mut self.pf.ensbl, &mut obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                                self.pf.errors = self.pf.obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_rmsep(obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();

                                PyObser(obser)
                            },
                            None => {
                                self.pf.model.simulate_ensbl(&mut self.pf.ensbl, &mut self.pf.obser, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                                self.pf.errors = self.pf.obser.par_ensbl_iter().chunks(64).map(|chunks| chunks.iter().map(|(_, out)| observec_rmsep(self.pf.obser.refdt(), out.as_slice())).collect::<Vec<Float>>()).flatten().collect::<Vec<Float>>();

                                PyObser(self.pf.obser.clone())
                            }
                        }
                    })
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _sir_mvllh>](&mut self, covm: &PyCovMatrix) -> PyResult<(Vec<Float>, Vec<usize>, Vec<Float>)> {
                    let llh = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        covm.0.observec_log_likelihood(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_sir_loop(&self.settings, &$simulate, &llh))
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _sir_iter_mvllh>](&mut self, covm: &PyCovMatrix) -> PyResult<(Float, usize, Float)> {
                    let llh = |o1: &[ObserVec<Float, $vdim>], o2: &[ObserVec<Float, $vdim>]| {
                        covm.0.observec_log_likelihood(o1, o2)
                    };

                    unroll_pf_errors!(self.pf.pf_sir(&self.settings, &$simulate, &llh))
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
            pub struct [<$name $rcount Model>](pub $model<Float, $rcount, MultivariateDensity<Float, $dim>>);}

            paste! {
            #[pymethods]
            impl [<$name $rcount Model>] {
                #[allow(missing_docs)]
                pub fn diagnostics(&self, scobs: &PyScObs, ensbl: &mut [<$name $rcount Ensbl>]) -> PyDiagObser {
                    self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                    let mut obser = Obser::<Float, ICSCoordsBasis<Float>>::new(scobs.0.as_scobs(), ensbl.0.len());

                    self.0.simulate_ics_basis_ensbl(&mut ensbl.0, &mut obser).unwrap();

                    PyDiagObser(obser)
                }

                #[allow(missing_docs)]
                pub fn limit_latitude(&mut self, max_lat: Float) {
                    self.0.limit_latitude(max_lat)
                }

                #[allow(missing_docs)]
                #[new]
                pub fn new(priors: Vec<PyUnivariate>, input: &PyWSAInputData, radial_resolution: Float) -> PyResult<Self> {
                    let names = $model::<Float, $rcount, MultivariateDensity<Float, $dim>>::PARAMS;

                    if priors.len() != $dim {
                        Err(PyValueError::new_err("invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.0.0 == *next.1)
                        }) {
                            let mvpdf = MultivariateDensity::new(priors.iter().map(|uvpdf| &uvpdf.0 .1));


                            Ok(Self($model::<Float, $rcount, MultivariateDensity<Float, $dim>>::new(mvpdf, input.0.clone(), radial_resolution)))
                        } else {
                            Err(PyValueError::new_err("invalid parameter names"))
                        }
                    }
                }

                #[allow(missing_docs)]
                #[pyo3(signature = (scobs, size, initial_seed, **opt_kwargs))]
                pub fn new_pf(&self, scobs: &PyScObs, size: usize, initial_seed: u64, opt_kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<[<$name $rcount ParticleFilter>]> {

                    Ok(match opt_kwargs {
                        Some(kwargs) => {
                            [<$name $rcount ParticleFilter>] {
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
                            [<$name $rcount ParticleFilter>] {
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
                pub fn [<$obsname _simulate>](&self, ensbl: &mut [<$name $rcount Ensbl>], obser: &mut [<$name $rcount Obser>])  {
                    self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                    self.0.simulate_ensbl(&mut ensbl.0, &mut obser.0, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                }
            }
        }
    };
}

impl_wsahux_model!(
    "WSAHUX",
    "pbv",
    models::WSAHUXModel,
    215,
    8,
    1,
    (),
    models::WSAState<Float>,
    models::WSAHUXModel::observe_pbv
);
