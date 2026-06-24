use crate::Float;
use bayesfm::{
    conf::ConfTime,
    geometry::{Geometry, Geometry3D},
    EnsembleModel,
};
use numpy::ToPyArray;
use ocnus::mag::Magnetometer;
use pyo3::types::PyAnyMethods;

macro_rules! impl_py_mfr_model {
    ($model: ty, $name: ident, $nparams: expr) => {
        paste::paste! {
            #[allow(clippy::upper_case_acronyms)]
            #[pyo3::pyclass]
            #[doc = "The " $name " model."]
            pub struct $name(pub $model<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>);

            bayesfm::py_add_model_functions!($model, $name, $nparams);
            bayesfm::py_add_model_simulation!($model, $name, $nparams, "mag3", BasicConf, 3, 3);
            bayesfm::py_add_model_filter!($model, $name, $nparams, "Mag3", BasicConf, 3, 3, 3);

            #[pyo3::pymethods]
            impl $name
            {
                /// Create a new atmosphere model from univariate prior distributions and shells.
                #[new]
                pub fn new(priors: Vec<crate::PyUnivariate>) -> pyo3::PyResult<Self> {
                    let names = $model::<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>::PARAM_NAMES;

                    if priors.len() != $nparams {
                        Err(pyo3::exceptions::PyValueError::new_err("Invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.name() == *next.1)
                        }) {
                            let mvpdf = prodef::MultivariateDensity::new(nalgebra::SVector::from_iterator(priors.iter().map(|uvpdf| uvpdf.density().clone())));

                            Ok(Self($model::<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>::new(mvpdf)))
                        } else {
                            Err(pyo3::exceptions::PyValueError::new_err("Invalid parameter names"))
                        }
                    }
                }

                /// Generate iso surfaces for a given model state ensemble, input array, and set of parameters.
                #[allow(clippy::too_many_arguments)]
                pub fn wireframe<'py>(
                    &self,
                    py: pyo3::Python<'py>,
                    input: numpy::PyReadonlyArray2<Float>,
                    initial: bayesfm::pytypes::PyBasicConf3,
                    snapshot: bayesfm::pytypes::PyBasicConf3,
                    mu: Float,
                    nus: numpy::PyReadonlyArray2<Float>,
                    ss: numpy::PyReadonlyArray2<Float>
                ) -> pyo3::PyResult<(pyo3::Bound<'py, numpy::PyArray2<Float>>, pyo3::Bound<'py, numpy::PyArray2<Float>>, pyo3::Bound<'py, numpy::PyArray2<Float>>)> {
                    let matrix = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Const<$nparams>, nalgebra::Dyn>(input, "input")?;
                    let nu_mat = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Dyn, nalgebra::Dyn>(nus, "nu")?;
                    let s_mat = bayesfm::pytypes::array_to_matrix::<numpy::ndarray::Dim<[usize; 2]>, nalgebra::Dyn, nalgebra::Dyn>(ss, "s")?;

                    let time_step = snapshot.0.timestamp() - initial.0.timestamp();

                   let isos =  py.detach(|| {
                        let mut model_ensbl = bayesfm::EnsembleState::new(matrix, None, None);

                        bayesfm::py_unroll_model_errors!(self.0.initialize_states_ensbl(&mut model_ensbl))?;
                        bayesfm::py_unroll_model_errors!(self.0.evolve_fmst_ensbl_par(time_step, &mut model_ensbl))?;

                        Ok(bayesfm::py_unwrap!($model::<Float, prodef::MultivariateDensity<Float, nalgebra::Const<$nparams>>>::iso_surface_mu::<$nparams, nalgebra::Const<1>, nalgebra::Const<$nparams>>(mu, nu_mat, s_mat, &model_ensbl.params().column(0), &model_ensbl.state(0).1), "failed to generate iso surfaces"))
                    })?;

                    Ok((isos[0].to_pyarray(py), isos[1].to_pyarray(py), isos[2].to_pyarray(py)))
                }
            }
        }
    };
}

impl_py_mfr_model!(ocnus::models::NC16Model, NC16, 9);
impl_py_mfr_model!(ocnus::models::CCLFFModel, CCLFF, 8);
impl_py_mfr_model!(ocnus::models::CCUTModel, CCUT, 8);
