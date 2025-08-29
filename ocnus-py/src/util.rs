use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix};
use numpy::{ndarray::Dimension, PyReadonlyArray};
use pyo3::{exceptions::PyValueError, PyResult};

use crate::Float;

pub fn array_to_matrix<D, R, C, RStride, CStride>(
    array: PyReadonlyArray<Float, D>,
) -> PyResult<OMatrix<Float, C, R>>
where
    D: Dimension,
    R: Dim,
    C: Dim,
    RStride: Dim,
    CStride: Dim,
    DefaultAllocator: Allocator<C, R>,
{
    match array.try_as_matrix::<R, C, RStride, CStride>() {
        Some(value) => Ok(value.transpose()),
        None => Err(PyValueError::new_err(
            "conversion of numpy array to nalgebra matrix failed",
        )),
    }
}
