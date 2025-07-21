use crate::Float;
use ocnus::stats::{
    ConstantDensity, CosineDensity, MultivariateDensity, NormalDensity, ReciprocalDensity,
    UniformDensity, UnivariateDensity,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[allow(missing_docs)]
#[pyclass(name = "Prior")]
pub struct PyPrior(pub Vec<PyUnivariate>);

impl PyPrior {
    /// Create a [`MultivariateDensity`] from self with a compile-time known dimensionality.
    pub fn as_multivarate_density<const D: usize>(&self) -> MultivariateDensity<Float, D> {
        MultivariateDensity::new(self.0.iter().map(|uvpdf| &uvpdf.0 .1))
    }
}

#[pymethods]
impl PyPrior {
    #[allow(missing_docs)]
    #[new]
    pub fn new(priors: Vec<PyUnivariate>) -> PyResult<Self> {
        Ok(Self(priors.clone()))
    }
}

#[allow(missing_docs)]
#[derive(Clone)]
#[pyclass(name = "Univariate")]
pub struct PyUnivariate(pub (String, UnivariateDensity<Float>));

#[pymethods]
impl PyUnivariate {
    #[allow(missing_docs)]
    #[classmethod]
    pub fn constant(_cls: &Bound<PyType>, name: String, value: Float) -> PyResult<Self> {
        Ok(Self((name, ConstantDensity::new(value))))
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn cosine(_cls: &Bound<PyType>, name: String, min: Float, max: Float) -> PyResult<Self> {
        match CosineDensity::new(min, max) {
            Some(value) => Ok(Self((name, value))),
            None => Err(PyValueError::new_err("invalid range")),
        }
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn normal(
        _cls: &Bound<PyType>,
        name: String,
        mean: Float,
        std_dev: Float,
        min: Float,
        max: Float,
    ) -> PyResult<Self> {
        match NormalDensity::new(mean, std_dev, min, max) {
            Some(value) => Ok(Self((name, value))),
            None => Err(PyValueError::new_err("invalid range")),
        }
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn reciprocal(
        _cls: &Bound<PyType>,
        name: String,
        min: Float,
        max: Float,
    ) -> PyResult<Self> {
        match ReciprocalDensity::new(min, max) {
            Some(value) => Ok(Self((name, value))),
            None => Err(PyValueError::new_err("invalid range")),
        }
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn uniform(_cls: &Bound<PyType>, name: String, min: Float, max: Float) -> PyResult<Self> {
        match UniformDensity::new(min, max) {
            Some(value) => Ok(Self((name, value))),
            None => Err(PyValueError::new_err("invalid range")),
        }
    }
}
