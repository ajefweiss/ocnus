//! Python module for the **ocnus** framework.

mod data;
mod filter;
mod frm;
mod geo;
mod macros;
mod noise;
mod obs;
mod swm;
mod util;

pub use data::*;
pub use frm::*;
pub use geo::*;
pub use noise::*;
pub use obs::*;
pub use swm::*;

use prodef_py::*;
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

    // ProDeF re-exports.
    m.add_class::<PyUnivariate>()?;
    m.add_class::<PyMultivariate>()?;

    // Observation types.
    m.add_class::<PyObs>()?;
    m.add_class::<PyObsType>()?;

    // Observation data types.
    m.add_class::<PyObsData>()?;
    m.add_class::<PyObsDataType>()?;

    // Noise model types.
    m.add_class::<PyObsVecNoise>()?;

    // Models.
    m.add_class::<AGCSModel>()?;
    m.add_class::<AtmosphereT2O2Model>()?;
    m.add_class::<AtmosphereT5O4Model>()?;
    m.add_class::<CCLFFModel>()?;
    m.add_class::<COREModel>()?;
    m.add_class::<CCUTModel>()?;
    m.add_class::<ECHModel>()?;
    m.add_class::<NC16Model>()?;
    m.add_class::<WSAHUX215Model>()?;

    // Model specific types.
    m.add_class::<PyWSAInputData>()?;

    // Filters.
    m.add_class::<AGCSMagFilter>()?;
    m.add_class::<AtmosphereT2O2DensityFilter>()?;
    m.add_class::<CCLFFMagFilter>()?;
    m.add_class::<COREMagFilter>()?;
    m.add_class::<CCUTMagFilter>()?;
    m.add_class::<ECHMagFilter>()?;
    m.add_class::<NC16MagFilter>()?;
    m.add_class::<WSAHUX215PBSFilter>()?;

    Ok(())
}
