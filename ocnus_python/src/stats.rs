use crate::io::ValueError;
use ocnus::stats::{Constant1D, Cosine1D, Normal1D, Reciprocal1D, Uniform1D, Univariate1D};
use pyo3::{prelude::*, types::PyType};

#[derive(Clone)]
#[pyclass]
pub struct Prior32(pub Vec<Univariate32>);

#[pymethods]
impl Prior32 {
    #[classmethod]
    #[new]
    pub fn constructor(_cls: &Bound<'_, PyType>, priors: Vec<Univariate32>) -> PyResult<Self> {
        Ok(Self(priors))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Prior64(pub Vec<Univariate32>);

#[pymethods]
impl Prior64 {
    #[classmethod]
    #[new]
    pub fn constructor(_cls: &Bound<'_, PyType>, priors: Vec<Univariate32>) -> PyResult<Self> {
        Ok(Self(priors))
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Univariate32(pub (String, Univariate1D<f32>));

#[pymethods]
impl Univariate32 {
    #[classmethod]
    pub fn constant(_cls: &Bound<'_, PyType>, name: String, value: f32) -> PyResult<Self> {
        Ok(Self((name, Constant1D::new(value))))
    }

    #[classmethod]
    pub fn cosine(_cls: &Bound<'_, PyType>, name: String, range: (f32, f32)) -> PyResult<Self> {
        match Cosine1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn normal(
        _cls: &Bound<'_, PyType>,
        name: String,
        mean: f32,
        std_dev: f32,
        range: (f32, f32),
    ) -> PyResult<Self> {
        match Normal1D::new(mean, std_dev, range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn reciprocal(_cls: &Bound<'_, PyType>, name: String, range: (f32, f32)) -> PyResult<Self> {
        match Reciprocal1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn uniform(_cls: &Bound<'_, PyType>, name: String, range: (f32, f32)) -> PyResult<Self> {
        match Uniform1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }
}

#[derive(Clone)]
#[pyclass]
pub struct Univariate64(pub (String, Univariate1D<f64>));

#[pymethods]
impl Univariate64 {
    #[classmethod]
    pub fn constant(_cls: &Bound<'_, PyType>, name: String, value: f64) -> PyResult<Self> {
        Ok(Self((name, Constant1D::new(value))))
    }

    #[classmethod]
    pub fn cosine(_cls: &Bound<'_, PyType>, name: String, range: (f64, f64)) -> PyResult<Self> {
        match Cosine1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn normal(
        _cls: &Bound<'_, PyType>,
        name: String,
        mean: f64,
        std_dev: f64,
        range: (f64, f64),
    ) -> PyResult<Self> {
        match Normal1D::new(mean, std_dev, range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn reciprocal(_cls: &Bound<'_, PyType>, name: String, range: (f64, f64)) -> PyResult<Self> {
        match Reciprocal1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }

    #[classmethod]
    pub fn uniform(_cls: &Bound<'_, PyType>, name: String, range: (f64, f64)) -> PyResult<Self> {
        match Uniform1D::new(range) {
            Ok(value) => Ok(Self((name, value))),
            Err(error) => Err(ValueError::new_err(error.to_string())),
        }
    }
}
