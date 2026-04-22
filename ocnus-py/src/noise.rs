use crate::{Float, util::array_to_matrix};
use nalgebra::{Dyn, OVector};
use numpy::{PyReadonlyArray2, ndarray::Dim};
use ocnus::obs::noise::ObsVecNoise;
use prodef::{domain::UDomain, multinormal::MultiNormalDensity};
use pyo3::{Bound, exceptions::PyValueError, prelude::*, types::PyType};

#[pyclass(name = "ObsVecNoise")]
/// Observation vector noise model.
pub struct PyObsVecNoise(pub ObsVecNoise<Float>);

#[pymethods]
impl PyObsVecNoise {
    /// Create a new additive normal noise object using the standard deviation.
    #[classmethod]
    #[pyo3(signature = (std_dev, seed = 42))]
    fn additive_normal(_cls: &Bound<PyType>, std_dev: Float, seed: u64) -> PyResult<Self> {
        if std_dev < 0.0 || !std_dev.is_finite() {
            Err(PyValueError::new_err(
                "The normal distribution standard deviation must be a positive number",
            ))
        } else {
            Ok(Self(ObsVecNoise::AdditiveNormal(std_dev, seed)))
        }
    }

    /// Create a new additive multinormal noise object using a covariance matrix.
    #[classmethod]
    #[pyo3(signature = (covariance, seed = 42))]
    fn additive_multinormal(
        _cls: &Bound<PyType>,
        covariance: PyReadonlyArray2<Float>,
        seed: u64,
    ) -> PyResult<Self> {
        let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(covariance)?;
        let length = matrix.nrows();

        let mvnpdf = MultiNormalDensity::from_matrix(
            matrix,
            OVector::<Float, Dyn>::zeros(length),
            UDomain::new(Dyn(length)),
        )
        .unwrap();

        Ok(Self(ObsVecNoise::AdditiveMultiNormal(mvnpdf, seed)))
    }

    /// Create a new multiplicative normal noise object using the standard deviation.
    #[classmethod]
    #[pyo3(signature = (std_dev, seed = 42))]
    fn multiplicative_normal(_cls: &Bound<PyType>, std_dev: Float, seed: u64) -> PyResult<Self> {
        if std_dev < 0.0 || !std_dev.is_finite() {
            Err(PyValueError::new_err(
                "The normal distribution standard deviation must be a positive number",
            ))
        } else {
            Ok(Self(ObsVecNoise::MultiplicativeNormal(std_dev, seed)))
        }
    }
}
