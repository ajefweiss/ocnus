use crate::{
    AGCSModel, COREModel, Float, PyObs, PyObsData, frm::unroll_model_errors, util::array_to_matrix,
};
use nalgebra::Dyn;
use numpy::{PyReadonlyArray2, ndarray::Dim};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError},
    instr::{Plasma, WLCamera},
    obs::{ObsEnsbl, noise::NullNoise},
};
use paste::paste;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

macro_rules! impl_plasma_model {
    ($name: literal, $model: ty, $params: expr, $csst: ty, $fmst: ty) => {
        paste! {
            #[pymethods]
            impl [<$name Model>] {
                /// Simulate proton density for an observation configuration and input array.
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

                /// Simulate a remote white light image for an observation configuration and input array.
                pub fn simulate_rwl<'py>(&self, py: Python<'py>, obs: &PyObs, input: PyReadonlyArray2<Float>) -> PyResult<PyObsData> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;

                    py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);

                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;

                        let mut obs_ensbl = ObsEnsbl::new(obs.as_cam()?.clone(), model_ensbl.len(), None).unwrap();

                        unroll_model_errors!(self.0.simulate_ensbl(&mut model_ensbl, &mut obs_ensbl, &$model::observe_rwl,  &mut None::<&mut NullNoise<Float>>))?;

                        Ok(obs_ensbl.clone().into())
                    })
                }
            }
        }
    };
}

impl_plasma_model!(
    "AGCS",
    ocnus_frm::models::AGCSModel,
    13,
    AGCSState<Float>,
    COREState<T>
);

impl_plasma_model!(
    "CORE",
    ocnus_frm::models::COREModel,
    11,
    XTState<Float>,
    COREState<T>
);
