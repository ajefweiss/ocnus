#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

mod covm;
mod models;
mod noise;
mod obser;
mod stats;

pub use covm::*;
pub use models::*;
pub use noise::*;
pub use obser::*;
pub use stats::*;

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

    m.add_class::<PyCovMatrix>()?;
    m.add_class::<PyPrior>()?;
    m.add_class::<PyObserVecNoise>()?;
    m.add_class::<PyUnivariate>()?;
    m.add_class::<PyObser>()?;
    m.add_class::<PyMagScObs>()?;
    m.add_class::<PyScObs>()?;
    m.add_class::<PyMagScObs>()?;
    m.add_class::<PyWSAInputData>()?;

    m.add_class::<NC16Ensbl>()?;
    m.add_class::<NC16Model>()?;
    m.add_class::<NC16Obser>()?;
    m.add_class::<NC16ParticleFilter>()?;

    m.add_class::<ECHEnsbl>()?;
    m.add_class::<ECHModel>()?;
    m.add_class::<ECHObser>()?;
    m.add_class::<ECHParticleFilter>()?;

    m.add_class::<COREEnsbl>()?;
    m.add_class::<COREModel>()?;
    m.add_class::<COREObser>()?;
    m.add_class::<COREParticleFilter>()?;

    m.add_class::<WSAHUX215Ensbl>()?;
    m.add_class::<WSAHUX215Model>()?;
    m.add_class::<WSAHUX215Obser>()?;
    m.add_class::<WSAHUX215ParticleFilter>()?;

    Ok(())
}
