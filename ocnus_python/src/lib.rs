pub mod io;
mod models;
mod obser;
mod stats;

use models::CORE32Ensbl;
use stats::{Prior32, Prior64, Univariate32, Univariate64};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn ocnus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    m.add_class::<CORE32Ensbl>()?;

    m.add_class::<Prior32>()?;
    m.add_class::<Prior64>()?;
    m.add_class::<Univariate32>()?;
    m.add_class::<Univariate64>()?;

    Ok(())
}
