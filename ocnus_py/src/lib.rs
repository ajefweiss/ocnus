//! Python module for **contigo-rs**.

mod models;

// Configure floating-point type, by default we use f64 as this is the default in python.
#[cfg(not(feature = "f32"))]
/// 64-bit floating point type.
pub type Float = f64;
#[cfg(feature = "f32")]
/// 32-bit floating point type.
pub type Float = f32;

#[pyo3::pymodule]
mod ocnus_py {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        pyo3_log::init();
        Ok(())
    }

    // Re-export external components directly into Python
    #[pymodule_export]
    use prodef::pytypes::{PyMultivariate, PyUnivariate};

    #[pymodule_export]
    use bayesfm::pytypes::{
        PyEnsblLocation3ObsVec3, PyEnsblWCSConfImg, PyLocation3, PyLocation3Series, PyObsVecNoise,
        PyWCSConf, PyWCSConfSeries, PyXoshiro256PlusPlus,
    };

    #[pymodule_export]
    use crate::models::{AGCS, CCLFF, CCUT, CORE, NC16};
}
