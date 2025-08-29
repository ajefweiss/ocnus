use crate::{util::array_to_matrix, Float};
use nalgebra::Dyn;
use numpy::{ndarray::Dim, PyReadonlyArray2};
use ocnus::math::CovMatrix;
use pyo3::{exceptions::PyValueError, prelude::*};

#[allow(missing_docs)]
#[pyclass(name = "CovMatrix")]
pub struct PyCovMatrix(pub CovMatrix<Float, Dyn>);

#[pymethods]
impl PyCovMatrix {
    #[new]
    /// Create a new [`PyCovMatrix`] from a 2-dimensional NumPy array.
    fn new(array: PyReadonlyArray2<Float>) -> PyResult<Self> {
        let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(array)?;

        let result = ocnus::math::CovMatrix::new(matrix, true);

        match result {
            Some(covmatrix) => Ok(Self(covmatrix)),
            None => Err(PyValueError::new_err("invalid covariance matrix")),
        }
    }
}
