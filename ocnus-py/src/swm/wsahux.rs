use crate::{
    Float, PyICSBasis3, PyObs, PyObsData, PyObsVecNoise, PyUnivariate,
    filter::impl_filter_obsvec,
    macros::{select_error_metric, unroll_filter_errors, unroll_model_errors},
    util::{array_to_matrix, py_any_iterator, py_unwrap},
};
use nalgebra::{Const, DMatrix, DVector, Dyn, OMatrix, SVector, U1};
use numpy::{
    PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray, ndarray::Dim,
};
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
use ocnus_swm::models::WSAInputData;
use paste::paste;
use prodef::{domain::UDomain, multinormal::MultiNormalDensity, multivariate::MultivariateDensity};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyDict, PyString, PyType},
};

#[pyclass(name = "WSAInputData")]
/// Input data for WSAHUX model.
pub struct PyWSAInputData(pub WSAInputData<Float>);

#[pymethods]
impl PyWSAInputData {
    #[classmethod]
    /// Load WSA input data from a JSON5 file.
    pub fn load(_cls: &Bound<PyType>, path: &Bound<PyString>) -> PyResult<Self> {
        match serde_json5::from_str::<WSAInputData<Float>>(&std::fs::read_to_string(
            path.extract::<String>().unwrap(),
        )?) {
            Ok(value) => Ok(Self(value)),
            Err(_) => Err(PyValueError::new_err("failed to load from file")),
        }
    }

    #[new]
    #[pyo3(signature = (dmap, efs, opt_lon_1d = None, opt_lat_1d = None, opt_lat_indices = None))]
    /// Create WSA input data object from numpy arrays.
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

macro_rules! impl_wsahux_model {
    ($name: literal, $model: ty, $rcount: expr, $params: expr) => {
        paste! {
            #[derive(Clone)]
            #[doc = "Filter for " $name $rcount " model with PBS observations."]
            #[pyclass(from_py_object)]
            pub struct [<$name $rcount PBS Filter>] (
                FilterObject<Float, VecConf<Float, 3>, ObsVec<Float, 1>, $model<Float, $rcount, MultivariateDensity<Float, Const<$params>>>, 3, $params>,
            );
        }

        paste! {
            #[pyclass]
            #[doc = $name $rcount " model."]
            pub struct [<$name $rcount Model>](pub $model<Float, $rcount, MultivariateDensity<Float, Const<$params>>>);}

            paste! {
            #[pymethods]
            impl [<$name $rcount Model>] {
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
                    $model::<Float, $rcount, MultivariateDensity<Float, Const<$params>>>::PARAMS.iter().map(|name| name.to_string()).collect()
                }

                /// Limit the maximum latitude for the model.
                pub fn limit_latitude(&mut self, max_lat: Float) {
                    self.0.limit_latitude(max_lat)
                }

                /// Create a new WSAHUX model from univariate prior distributions and input data.
                #[new]
                pub fn new(priors: Vec<PyUnivariate>, input: &PyWSAInputData, radial_resolution: Float) -> PyResult<Self> {
                    let names = $model::<Float, $rcount, MultivariateDensity<Float, Const<$params>>>::PARAMS;

                    if priors.len() != $params {
                        Err(PyValueError::new_err("Invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.name() == *next.1)
                        }) {
                            let mvpdf = MultivariateDensity::new(SVector::from_iterator(priors.iter().map(|uvpdf| uvpdf.density().clone())));

                            Ok(Self($model::<Float, $rcount, MultivariateDensity<Float, Const<$params>>>::new(mvpdf, input.0.clone(), radial_resolution)))
                        } else {
                            Err(PyValueError::new_err("Invalid parameter names"))
                        }
                    }
                }

                /// Simulate plasma bulk speed for an observation configuration and input array.
                pub fn simulate_pbs<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);
                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let mut obs_ensbl = ObsEnsbl::new(obs.as_obs3()?.clone(), model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_ensbl(&mut model_ensbl, &mut obs_ensbl, &$model::observe_pbs,  &mut None::<&mut NullNoise<Float>>))?;

                        Ok(obs_ensbl.clone().into())
                    })
                }
            }
        }
    };
}

impl_wsahux_model!("WSAHUX", ocnus_swm::models::WSAHUXModel, 215, 8);
impl_filter_obsvec!(
    "WSAHUX215",
    "PBS",
    ocnus_swm::models::WSAHUXModel,
    ocnus_swm::models::WSAHUXModel::observe_pbs,
    3,
    1,
    8,
    (),
    WSAState<T>
);
