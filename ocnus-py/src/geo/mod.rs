use crate::{
    Float, PyICSBasis3, PyObs, PyObsData, PyObsVecNoise, PyUnivariate,
    filter::impl_filter_obsvec,
    macros::{select_error_metric, unroll_filter_errors, unroll_model_errors},
    util::{array_to_matrix, py_any_iterator, py_unwrap},
};
use nalgebra::{Const, DVector, Dyn, OMatrix, SVector, U1};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, ndarray::Dim};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError},
    coords::Coordinates,
    instr::Plasma,
    methods::filters::{FilterError, FilterObject},
    obs::{
        ObsEnsbl,
        conf::VecConf,
        data::{ICSBasis, ObsVec, ObsVecMetric, ov_error},
        noise::NullNoise,
    },
};
use paste::paste;
use prodef::{domain::UDomain, multinormal::MultiNormalDensity, multivariate::MultivariateDensity};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyType},
};

macro_rules! impl_atmosphere_model {
    ($name: literal, $model: ty, $params: expr) => {
        paste! {
            #[derive(Clone)]
            #[doc = "Filter for " $name " model with density observations."]
            #[pyclass(from_py_object)]
            pub struct [<$name DensityFilter>] (
                FilterObject<Float, VecConf<Float, 3>, ObsVec<Float, 1>, $model<Float, MultivariateDensity<Float, Const<$params>>>, 3, $params>,
            );
        }

        paste! {
            #[pyclass]
            #[doc = $name " model."]
            pub struct [<$name Model>](pub $model<Float, MultivariateDensity<Float, Const<$params>>>);
        }

        paste! {
            #[pymethods]
            impl [<$name Model>] {
                /// Simulate ICS basis observation ensemble from input model state ensemble.
                pub fn icsbasis<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);

                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let obs3 = obs.as_obs3()?.clone();

                        let mut obs_ensbl = ObsEnsbl::<Float, VecConf<Float, 3>, ICSBasis<Float, 3>>::new(obs3, model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_icsbasis_ensbl(&mut model_ensbl, &mut obs_ensbl))?;

                        Ok(PyObsData::B3(PyICSBasis3(obs_ensbl)))
                    })
                }

                /// Get the names of the model parameters.
                #[classmethod]
                pub fn names(_cls: &Bound<PyType>,) -> Vec<String> {
                    $model::<Float, MultivariateDensity<Float, Const<$params>>>::PARAMS.iter().map(|name| name.to_string()).collect()
                }

                /// Create a new atmosphere model from univariate prior distributions and input data.
                #[new]
                pub fn new(priors: Vec<PyUnivariate>) -> PyResult<Self> {
                    let names = $model::<Float, MultivariateDensity<Float, Const<$params>>>::PARAMS;

                    if priors.len() != $params {
                        Err(PyValueError::new_err("Invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.name() == *next.1)
                        }) {
                            let mvpdf = MultivariateDensity::new(SVector::from_iterator(priors.iter().map(|uvpdf| uvpdf.density().clone())));

                            Ok(Self($model::<Float, MultivariateDensity<Float, Const<$params>>>::new(mvpdf)))
                        } else {
                            Err(PyValueError::new_err("Invalid parameter names"))
                        }
                    }
                }

                /// Simulate mass density for an observation configuration and input array.
                pub fn simulate_rho<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);
                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let mut obs_ensbl = ObsEnsbl::new(obs.as_obs3()?.clone(), model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_ensbl(&mut model_ensbl, &mut obs_ensbl, &$model::observe_rho,  &mut None::<&mut NullNoise<Float>>))?;

                        Ok(obs_ensbl.clone().into())
                    })
                }

                /// Simulate temperature for an observation configuration and input array.
                pub fn simulate_temp<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);
                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let mut obs_ensbl = ObsEnsbl::new(obs.as_obs3()?.clone(), model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_ensbl(&mut model_ensbl, &mut obs_ensbl, &$model::observe_temp,  &mut None::<&mut NullNoise<Float>>))?;

                        Ok(obs_ensbl.clone().into())
                    })
                }
            }
        }
    };
}

impl_atmosphere_model!("AtmosphereT2O2", ocnus_geo::models::AtmosphereT2O2Model, 18);
impl_filter_obsvec!(
    "AtmosphereT2O2",
    "Density",
    ocnus_geo::models::AtmosphereT2O2Model,
    ocnus_geo::models::AtmosphereT2O2Model::observe_rho,
    3,
    1,
    18,
    (),
    ()
);

impl_atmosphere_model!("AtmosphereT5O4", ocnus_geo::models::AtmosphereT5O4Model, 53);
impl_filter_obsvec!(
    "AtmosphereT5O4",
    "Density",
    ocnus_geo::models::AtmosphereT5O4Model,
    ocnus_geo::models::AtmosphereT5O4Model::observe_rho,
    3,
    1,
    53,
    (),
    ()
);
