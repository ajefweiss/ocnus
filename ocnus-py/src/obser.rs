use crate::{util::array_to_matrix, Float};
use nalgebra::{DMatrix, Dyn, Vector3, U1};
use numpy::{ndarray::Dim, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use ocnus::{
    base::{Obser, ScConf, ScObs},
    obsty::{ICSCoordsBasis, ObserImg, ObserVec},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

#[allow(missing_docs)]
#[derive(Clone)]
#[pyclass(name = "DiagObser")]
pub struct PyDiagObser(pub Obser<Float, ICSCoordsBasis<Float>>);

#[pymethods]
impl PyDiagObser {
    #[allow(missing_docs)]
    pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
        let column = self.0.get_output(key);

        let matrix = DMatrix::from_iterator(
            12,
            self.0.len(),
            column.iter().flat_map(|icscb| icscb.iter().cloned()),
        );

        matrix.transpose().to_pyarray(py)
    }
}

#[allow(missing_docs)]
#[pyclass(name = "Obser")]
pub struct PyObser(pub Obser<Float, ObserVec<Float, 1>>);

#[pymethods]
impl PyObser {
    #[allow(missing_docs)]
    #[new]
    pub fn new(scobs: &PyScObs, size: usize) -> PyResult<Self> {
        Ok(Self(Obser::new(scobs.0.clone(), size)))
    }

    #[allow(missing_docs)]
    pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
        let column = self.0.get_output(key);

        let matrix = DMatrix::from_iterator(
            1,
            self.0.len(),
            column.iter().flat_map(|icscb| icscb.iter().cloned()),
        );

        matrix.transpose().to_pyarray(py)
    }
}

#[allow(missing_docs)]
#[pyclass(name = "MagObser")]
pub struct PyMagObser(pub Obser<Float, ObserVec<Float, 3>>);

#[pymethods]
impl PyMagObser {
    #[allow(missing_docs)]
    #[new]
    pub fn new(scobs: &PyMagScObs, size: usize) -> PyResult<Self> {
        Ok(Self(Obser::new(scobs.0.clone(), size)))
    }

    #[allow(missing_docs)]
    pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
        let column = self.0.get_output(key);

        let matrix = DMatrix::from_iterator(
            3,
            self.0.len(),
            column.iter().flat_map(|icscb| icscb.iter().cloned()),
        );

        matrix.transpose().to_pyarray(py)
    }
}

#[allow(missing_docs)]
#[pyclass(name = "ImageObser")]
pub struct PyImageObser(pub Obser<Float, ObserImg<Float>>);

#[pymethods]
impl PyImageObser {
    #[allow(missing_docs)]
    #[new]
    pub fn new(scobs: &PyScObs, size: usize) -> PyResult<Self> {
        Ok(Self(Obser::new(scobs.0.as_scobs(), size)))
    }

    #[allow(missing_docs)]
    pub fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: usize,
    ) -> Vec<Bound<'py, PyArray2<Float>>> {
        let column = self.0.get_output(key);

        Vec::from_iter(
            column
                .row_iter()
                .map(|image| image[(0, 0)].transpose().to_pyarray(py)),
        )
    }
}

#[allow(missing_docs)]
#[pyclass(name = "ScObs")]
pub struct PyScObs(pub ScObs<Float, ObserVec<Float, 1>>);

impl PyScObs {
    /// Create new [`ScObs`] with observables.
    pub fn as_scobs_with_observables(
        &self,
        data: &[ObserVec<Float, 1>],
    ) -> ScObs<Float, ObserVec<Float, 1>> {
        ScObs::from_iterator_with_observables(
            (&self.0)
                .into_iter()
                .zip(data)
                .map(|((timestamp, scconf), ot)| (*timestamp, scconf.clone(), ot.clone())),
        )
    }
}

#[pymethods]
impl PyScObs {
    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn combine(_cls: &Bound<PyType>, scobs_1: &Self, scobs_2: &Self) -> Self {
        let mut scobs = scobs_1.0.clone() + scobs_2.0.clone();

        scobs.sort_by_timestamp();

        Self(scobs)
    }

    #[allow(missing_docs)]
    #[new]
    #[pyo3(signature = (timestamps, position, opt_data = None))]
    fn new(
        timestamps: Vec<Float>,
        position: &Bound<PyAny>,
        opt_data: Option<PyReadonlyArray1<Float>>,
    ) -> PyResult<Self> {
        // Convert position(s) into appropriate data type.
        let iter = match position.try_iter() {
            Ok(value) => value.map(|py_obj| {
                ScConf::PositionViewport((
                    Vector3::from(py_obj.unwrap().extract::<[Float; 3]>().unwrap()),
                    [
                        Vector3::from([0.0, 0.0, 0.0]),
                        Vector3::from([0.0, -0.5, -0.5]),
                        Vector3::from([0.0, 0.5, -0.5]),
                    ],
                    (512, 1024 * 1024),
                ))
            }),
            Err(..) => return Err(PyValueError::new_err("position argument must be iterable")),
        };

        // let iter = match position.try_iter() {
        //     Ok(value) => value.map(|py_obj| {
        //         ScConf::Position(Vector3::from(
        //             py_obj.unwrap().extract::<[Float; 3]>().unwrap(),
        //         ))
        //     }),
        //     Err(..) => return Err(PyValueError::new_err("position argument must be iterable")),
        // };

        match opt_data {
            Some(data) => {
                let matrix = array_to_matrix::<Dim<[usize; 1]>, Dyn, U1, U1, Dyn>(data)?;

                let observables = matrix
                    .column_iter()
                    .map(|row| ObserVec::from(row.as_slice()))
                    .collect::<Vec<ObserVec<Float, 1>>>();

                Ok(Self(ScObs::from((timestamps, iter, observables))))
            }
            None => Ok(Self(ScObs::from((timestamps, iter)))),
        }
    }

    #[allow(missing_docs)]
    pub fn timestamps(&self) -> Vec<Float> {
        self.0.timestamps()
    }
}

#[allow(missing_docs)]
#[pyclass(name = "MagScObs")]
pub struct PyMagScObs(pub ScObs<Float, ObserVec<Float, 3>>);

impl PyMagScObs {
    /// Create new [`ScObs`] with observables.
    pub fn as_scobs_with_observables(
        &self,
        data: &[ObserVec<Float, 3>],
    ) -> ScObs<Float, ObserVec<Float, 3>> {
        ScObs::from_iterator_with_observables(
            (&self.0)
                .into_iter()
                .zip(data)
                .map(|((timestamp, scconf), ot)| (*timestamp, scconf.clone(), ot.clone())),
        )
    }
}

#[pymethods]
impl PyMagScObs {
    fn __len__(&self) -> usize {
        self.0.len()
    }

    #[allow(missing_docs)]
    #[classmethod]
    pub fn combine(_cls: &Bound<PyType>, scobs_1: &Self, scobs_2: &Self) -> Self {
        let mut scobs = scobs_1.0.clone() + scobs_2.0.clone();

        scobs.sort_by_timestamp();

        Self(scobs)
    }

    #[allow(missing_docs)]
    #[new]
    #[pyo3(signature = (timestamps, position, opt_data = None))]
    fn new(
        timestamps: Vec<Float>,
        position: &Bound<PyAny>,
        opt_data: Option<PyReadonlyArray2<Float>>,
    ) -> PyResult<Self> {
        // Convert position(s) into appropriate data type.
        let iter = match position.try_iter() {
            Ok(value) => value.map(|py_obj| {
                ScConf::Position(Vector3::from(
                    py_obj.unwrap().extract::<[Float; 3]>().unwrap(),
                ))
            }),
            Err(..) => return Err(PyValueError::new_err("position argument must be iterable")),
        };

        match opt_data {
            Some(data) => {
                let matrix = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(data)?;

                let observables = matrix
                    .column_iter()
                    .map(|row| ObserVec::from(row.as_slice()))
                    .collect::<Vec<ObserVec<Float, 3>>>();

                Ok(Self(ScObs::from((timestamps, iter, observables))))
            }
            None => Ok(Self(ScObs::from((timestamps, iter)))),
        }
    }

    #[allow(missing_docs)]
    pub fn timestamps(&self) -> Vec<Float> {
        self.0.timestamps()
    }
}
