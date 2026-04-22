// mod filter;
mod mag;
mod plasma;

pub use mag::*;

use crate::{
    Float, PyObs, PyObsData, PyUnivariate,
    macros::unroll_model_errors,
    util::{array_to_matrix, py_unwrap},
};
use nalgebra::{Const, Dyn, SVector};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, ndarray::Dim};
use ocnus::{
    base::{Model, ModelEnsbl, ModelError},
    coords::{Coordinates, Coordinates3D},
    obs::{ObsEnsbl, conf::VecConf, data::ICSBasis},
};

use paste::paste;
use prodef::multivariate::MultivariateDensity;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::PyType,
};

macro_rules! impl_model3 {
    ($name: literal, $model: ty, $params: expr) => {
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

                        Ok(obs_ensbl.into())
                    })
                }

                /// Get the names of the model parameters.
                #[classmethod]
                pub fn names(_cls: &Bound<PyType>,) -> Vec<String> {
                    $model::<Float, MultivariateDensity<Float, Const<$params>>>::PARAMS.iter().map(|name| name.to_string()).collect()
                }

                /// Create a new model from univariate prior distributions.
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


                /// Generate iso surfaces for a given model state ensemble, input array, and set of parameters.
                pub fn wireframe<'py>(&self, py: Python<'py>, input: PyReadonlyArray2<Float>, timestamp: Float, mu: Float, nus: PyReadonlyArray2<Float>, ss: PyReadonlyArray2<Float>) -> PyResult<(Bound<'py, PyArray2<Float>>, Bound<'py, PyArray2<Float>>, Bound<'py, PyArray2<Float>>)> {
                    let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(input)?;
                    let nu_mat = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(nus)?;
                    let s_mat = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(ss)?;

                   let isos =  py.detach(|| {
                        let mut model_ensbl = ModelEnsbl::from_view(matrix.as_view(), None, None);

                        unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;
                        unroll_model_errors!(self.0.evolve_ensbl(timestamp, &mut model_ensbl))?;

                        Ok(py_unwrap!($model::<Float, MultivariateDensity<Float, Const<$params>>>::iso_surface_mu::<8, Const<1>, Const<$params>>(mu, nu_mat, s_mat, &model_ensbl.input.as_view(), &model_ensbl.state(0).1), "failed to generate iso surfaces"))
                    })?;

                    Ok((isos[0].to_pyarray(py), isos[1].to_pyarray(py), isos[2].to_pyarray(py)))
                }

                // TODO: implement wireframe
                // pub fn wireframe<'py>(&self, py: Python<'py>, index: usize, mu: Float, nus: PyReadonlyArray2<Float>, ss: PyReadonlyArray2<Float>) -> PyResult<(Bound<'py, PyArray2<Float>>, Bound<'py, PyArray2<Float>>, Bound<'py, PyArray2<Float>>)> {
                //     let params = py_unwrap!(self.0.ptpdf.particle(index), "index out of range");
                //     let state = &self.0.cs_states[index];

                //     let nu_mat = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(nus)?;
                //     let s_mat = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(ss)?;

                //     let isos = py_unwrap!($model::<Float, MultivariateDensity<Float, $params>>::iso_surface_mu::<8, Const<1>, Const<$params>>(mu, nu_mat, s_mat, &params, &state), "failed to generate iso surfaces");

                //     Ok((isos[0].to_pyarray(py), isos[1].to_pyarray(py), isos[2].to_pyarray(py)))
                // }
            }
        }
    };
}

impl_model3!("AGCS", ocnus_frm::models::AGCSModel, 13);
impl_model3!("CCLFF", ocnus_frm::models::CCLFFModel, 8);
impl_model3!("CORE", ocnus_frm::models::COREModel, 11);
impl_model3!("CCUT", ocnus_frm::models::CCUTModel, 8);
impl_model3!("ECH", ocnus_frm::models::ECHModel, 12);
impl_model3!("NC16", ocnus_frm::models::NC16Model, 9);
