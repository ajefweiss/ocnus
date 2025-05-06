use ocnus::{coords::TTState, models::COREState, OcnusEnsbl};
use pyo3::prelude::*;

#[pyclass]
pub struct CORE32Ensbl(pub OcnusEnsbl<f32, 11, COREState<f32>, TTState<f32>>);

// #[pymethods]
// impl CORE32Ensbl {
//     #[classmethod]
//     #[new]
//     pub fn initialize
// }
