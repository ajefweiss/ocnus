//! Python module for **contigo-rs**.

mod models;

use bayesfm::pytypes::*;
use models::*;
use prodef::pytypes::*;
use pyo3::prelude::*;

// Configure floating-point type, by default we use f64 as this is the default in python.
#[cfg(not(feature = "f32"))]
/// 64-bit floating point type.
pub type Float = f64;
#[cfg(feature = "f32")]
/// 32-bit floating point type.
pub type Float = f32;

#[pymodule]
fn ocnus_py(m: &Bound<PyModule>) -> PyResult<()> {
    pyo3_log::init();

    // prodef-rs re-exports
    m.add_class::<PyUnivariate>()?;
    m.add_class::<PyMultivariate>()?;

    // bayesfm-rs re-exports
    m.add_class::<PyLocation3>()?;
    m.add_class::<PyLocation3Series>()?;
    m.add_class::<PyWCSConf>()?;
    m.add_class::<PyWCSConfSeries>()?;
    m.add_class::<PyEnsblLocation3ObsVec3>()?;
    m.add_class::<PyEnsblWCSConfImg>()?;
    m.add_class::<PyObsVecNoise>()?;

    // ocnus models
    m.add_class::<NC16>()?;
    m.add_class::<CCUT>()?;
    m.add_class::<CCLFF>()?;
    m.add_class::<CORE>()?;
    m.add_class::<AGCS>()?;

    Ok(())
}
