use nalgebra::{DefaultAllocator, Dim, OMatrix, allocator::Allocator};
use numpy::{PyReadonlyArray, ndarray::Dimension};
use pyo3::{PyResult, exceptions::PyValueError};

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
            "Conversion of a numpy array to nalgebra matrix failed",
        )),
    }
}

macro_rules! py_any_iterator {
    ($iterator: expr, $type: ty) => {
        match $iterator.try_iter() {
            Ok(value) => value.map(|py_obj| py_obj.unwrap().extract::<$type>().unwrap()),
            Err(..) => return Err(PyValueError::new_err("Values argument must be iterable")),
        }
    };
}

macro_rules! py_unwrap {
    ($expr: expr, $text: literal) => {
        match $expr {
            Some(value) => value,
            None => return Err(PyValueError::new_err($text)),
        }
    };
}

pub(crate) use py_any_iterator;
pub(crate) use py_unwrap;
