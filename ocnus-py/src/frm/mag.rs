use crate::{
    AGCSModel, CCLFFModel, CCUTModel, COREModel, ECHModel, Float, NC16Model, PyObs, PyObsData,
    PyObsVecNoise,
    filter::impl_filter_obsvec,
    macros::unroll_model_errors,
    macros::{select_error_metric, unroll_filter_errors},
    util::{array_to_matrix, py_any_iterator, py_unwrap},
};
use nalgebra::{Const, Dyn, SVector, U1};
use nalgebra::{DVector, OMatrix};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, ndarray::Dim};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError},
    instr::Magnetometer,
    methods::filters::{FilterError, FilterObject},
    obs::{
        ObsEnsbl,
        conf::VecConf,
        data::{ObsVec, ObsVecMetric, ov_error},
        noise::NullNoise,
    },
};
use paste::paste;
use prodef::{domain::UDomain, multinormal::MultiNormalDensity, multivariate::MultivariateDensity};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};

/// Implement python types for coronal rope ejection models.
macro_rules! impl_mag_model {
    ($name: literal, $model: ty, $obs: expr, $params: expr, $csst: ty, $fmst: ty) => {
        paste! {
            #[pymethods]
            impl [<$name Model>] {
                /// Compute the fisher information matrix for a  observation configuration and a specific set of model parameters with a given covariance.
                pub fn fisher_mag<'py>(&self, py: Python<'py>, obs: &PyObs, values: &Bound<PyAny>, covariance: PyReadonlyArray2<Float>) -> PyResult<Bound<'py, PyArray2<Float>>> {
                    let iter = py_any_iterator!(values, Float);
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(covariance)?;

                    let vector = SVector::<Float, $params>::from_iterator(iter);

                    let fisher = py.detach(|| {
                        let obs3 = obs.as_obs3()?.clone();

                        Ok::<_, PyErr>(self.0.fisher_mag(&obs3, &vector.as_view::<Const<$params>, U1, U1, Const<$params>>(), &matrix).unwrap())
                    })?;

                    Ok(fisher.transpose().to_pyarray(py))
                }

                /// Simulate magnetic field for an observation configuration and input array.
                pub fn simulate_mag<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);

                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let mut obs_ensbl = ObsEnsbl::new(obs.as_obs3()?.clone(), model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_ensbl(&mut model_ensbl, &mut obs_ensbl, &$model::observe_mag3,  &mut None::<&mut NullNoise<Float>>))?;

                        Ok(obs_ensbl.into())
                    })
                }
            }
        }

        paste! {
            #[derive(Clone)]
            #[pyclass(from_py_object)]
            #[doc = "Filter for " $name " model with Mag observations."]
            pub struct [<$name Mag Filter>] (
                 FilterObject<Float, VecConf<Float, 3>, ObsVec<Float, 3>, $model<Float, MultivariateDensity<Float, Const<$params>>>, 3, $params>,
            );
        }

        impl_filter_obsvec!(
            $name,
            "Mag",
            $model,
            $obs,
            3,
            3,
            $params,
            $csst,
            $fmst
        );
    };
}

impl_mag_model!(
    "AGCS",
    ocnus_frm::models::AGCSModel,
    ocnus_frm::models::AGCSModel::observe_mag3,
    13,
    AGCSState<Float>,
    COREState<T>
);
impl_mag_model!(
    "CCLFF",
    ocnus_frm::models::CCLFFModel,
    ocnus_frm::models::CCLFFModel::observe_mag3,
    8,
    XCState<Float>,
    ()
);
impl_mag_model!(
    "CORE",
    ocnus_frm::models::COREModel,
    ocnus_frm::models::COREModel::observe_mag3,
    11,
    XTState<Float>,
    COREState<T>
);
impl_mag_model!(
    "CCUT",
    ocnus_frm::models::CCUTModel,
    ocnus_frm::models::CCUTModel::observe_mag3,
    8,
    XCState<Float>,
    ()
);
impl_mag_model!(
    "ECH",
    ocnus_frm::models::ECHModel,
    ocnus_frm::models::ECHModel::observe_mag3,
    12,
    XCState<Float>,
    ()
);
impl_mag_model!(
    "NC16",
    ocnus_frm::models::NC16Model,
    ocnus_frm::models::NC16Model::observe_mag3,
    9,
    XCState<Float>,
    ()
);
