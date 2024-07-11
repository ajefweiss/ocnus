use pyo3::prelude::*;

#[pymodule]
fn _ocnus(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(rust_function, m)?)?;
    Ok(())
}

