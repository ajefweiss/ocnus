use crate::Float;
use nalgebra::{DMatrix, Dyn, Vector3, U1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use ocnus::{
    base::{Obser, ScConf, ScObs},
    obsty::{ICSCoordsBasis, ObserVec},
};
use pyo3::{prelude::*, types::PyType};

#[allow(missing_docs)]
#[derive(Clone)]
#[pyclass(name = "DiagObser")]
pub struct PyDiagObser(pub Obser<Float, ICSCoordsBasis<Float>>);

#[pymethods]
impl PyDiagObser {
    #[allow(missing_docs)]
    #[new]
    pub fn new(scobs: &PyMagScObs, size: usize) -> PyResult<Self> {
        Ok(Self(Obser::new(scobs.0.as_scobs(), size)))
    }

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
        Ok(Self(Obser::new(scobs.0.as_scobs(), size)))
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
        Ok(Self(Obser::new(scobs.0.as_scobs(), size)))
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
        position: [Float; 3],
        opt_data: Option<PyReadonlyArray1<Float>>,
    ) -> Self {
        match opt_data {
            Some(data) => {
                let matrix = data
                    .try_as_matrix::<Dyn, U1, U1, Dyn>()
                    .expect("failed to convert numpy array to matrix")
                    .transpose();
                let observables = matrix
                    .column_iter()
                    .map(|row| ObserVec::from(row.as_slice()))
                    .collect::<Vec<ObserVec<Float, 1>>>();

                Self(ScObs::from((
                    timestamps,
                    ScConf::Position(Vector3::from(position)),
                    observables,
                )))
            }
            None => Self(ScObs::from((
                timestamps,
                ScConf::Position(Vector3::from(position)),
            ))),
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
    ) -> Self {
        // Convert position(s) into appropriate data type.
        let iter = position
            .try_iter()
            .expect("values must be iterable")
            .map(|py_obj| {
                ScConf::Position(Vector3::from(
                    py_obj.unwrap().extract::<[Float; 3]>().unwrap(),
                ))
            });

        match opt_data {
            Some(data) => {
                let matrix = data
                    .try_as_matrix::<Dyn, Dyn, Dyn, Dyn>()
                    .expect("failed to convert numpy array to matrix")
                    .transpose();
                let observables = matrix
                    .column_iter()
                    .map(|row| ObserVec::from(row.as_slice()))
                    .collect::<Vec<ObserVec<Float, 3>>>();

                Self(ScObs::from((timestamps, iter, observables)))
            }
            None => Self(ScObs::from((timestamps, iter))),
        }
    }

    #[allow(missing_docs)]
    pub fn timestamps(&self) -> Vec<Float> {
        self.0.timestamps()
    }
}
