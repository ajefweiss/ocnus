use crate::Float;
use ocnus::obsty::ObserVecNoise;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[allow(missing_docs)]
#[pyclass(name = "ObserVecNoise")]
pub struct PyObserVecNoise(pub ObserVecNoise<Float>);

#[pymethods]
impl PyObserVecNoise {
    /// Create a new gaussian noise object using the standard deviation.
    #[classmethod]
    #[pyo3(signature = (std_dev, seed = 42))]
    fn gaussian(_cls: &Bound<PyType>, std_dev: Float, seed: u64) -> PyResult<Self> {
        if std_dev < 0.0 || !std_dev.is_finite() {
            Err(PyValueError::new_err(
                "standard deviation must be a positive number",
            ))
        } else {
            Ok(Self(ObserVecNoise::Gaussian(std_dev, seed)))
        }
    }
}
