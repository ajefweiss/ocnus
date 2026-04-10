use crate::{
    Float, PyObs,
    util::{array_to_matrix, py_any_iterator, py_unwrap},
};
use nalgebra::{DMatrix, DVector, Dyn};
use numpy::{PyReadonlyArray2, ToPyArray, ndarray::Dim};
use ocnus::obs::{
    ObsEnsbl,
    conf::{CamConf, VecConf},
    data::{ICSBasis, ObsImg, ObsVec},
};
use paste::paste;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};

#[allow(missing_docs)]
#[derive(Clone)]
#[pyclass(from_py_object, name = "ObsData")]
pub enum PyObsData {
    B3(PyICSBasis3),
    P3IMG(PyObsP3IMG),
    P3V1(PyObsP3V1),
    P3V3(PyObsP3V3),
    P3V4(PyObsP3V4),
}

macro_rules! impl_py_ObsVec {
    ($ndim: expr, $values: expr, $name: literal) => {
        paste! {

            #[derive(Clone)]
            #[doc="PyObsData for n=" $ndim " and v="  $values]
            #[pyclass(from_py_object, name = $name)]
            pub struct [<PyObsP $ndim V $values>](pub ObsEnsbl<Float, VecConf<Float, $ndim>, ObsVec<Float, $values>>);
        }
    };
}

impl_py_ObsVec!(3, 1, "P3V1");
impl_py_ObsVec!(3, 3, "P3V3");
impl_py_ObsVec!(3, 4, "P3V4");

#[derive(Clone)]
#[pyclass(from_py_object, name = "ICSBasis")]
/// PyObsData for ICSBasis in 3D.
pub struct PyICSBasis3(pub ObsEnsbl<Float, VecConf<Float, 3>, ICSBasis<Float, 3>>);

#[derive(Clone)]
#[pyclass(from_py_object, name = "P3IMG")]
/// PyObsData for image observations in 3D.
pub struct PyObsP3IMG(pub ObsEnsbl<Float, CamConf<Float>, ObsImg<Float>>);

#[allow(missing_docs)]
#[derive(Clone)]
#[pyclass(from_py_object, name = "ObsDataType")]
pub enum PyObsDataType {
    B3,
    P3IMG,
    P3V1,
    P3V3,
    P3V4,
}

impl From<ObsEnsbl<Float, VecConf<Float, 3>, ICSBasis<Float, 3>>> for PyObsData {
    fn from(obs_ensbl: ObsEnsbl<Float, VecConf<Float, 3>, ICSBasis<Float, 3>>) -> Self {
        Self::B3(PyICSBasis3(obs_ensbl))
    }
}

impl From<ObsEnsbl<Float, CamConf<Float>, ObsImg<Float>>> for PyObsData {
    fn from(obs_ensbl: ObsEnsbl<Float, CamConf<Float>, ObsImg<Float>>) -> Self {
        Self::P3IMG(PyObsP3IMG(obs_ensbl))
    }
}

impl From<ObsEnsbl<Float, VecConf<Float, 3>, ObsVec<Float, 1>>> for PyObsData {
    fn from(obs_ensbl: ObsEnsbl<Float, VecConf<Float, 3>, ObsVec<Float, 1>>) -> Self {
        Self::P3V1(PyObsP3V1(obs_ensbl))
    }
}

impl From<ObsEnsbl<Float, VecConf<Float, 3>, ObsVec<Float, 3>>> for PyObsData {
    fn from(obs_ensbl: ObsEnsbl<Float, VecConf<Float, 3>, ObsVec<Float, 3>>) -> Self {
        Self::P3V3(PyObsP3V3(obs_ensbl))
    }
}

#[pymethods]
impl PyObsData {
    /// Return the observation data for a given ensemble member as a numpy array.
    pub fn get<'py>(&self, py: Python<'py>, key: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self {
            PyObsData::B3(data) => {
                let column = data.0.output(key);

                let matrix_ics = DMatrix::from_iterator(
                    3,
                    data.0.len(),
                    column.iter().flat_map(|icscb| icscb.ics().iter().cloned()),
                );

                let matrix_eps_mu = DMatrix::from_iterator(
                    3,
                    data.0.len(),
                    column
                        .iter()
                        .flat_map(|icscb| icscb.eps_mu().iter().cloned().collect::<Vec<Float>>()),
                );

                let matrix_eps_nu = DMatrix::from_iterator(
                    3,
                    data.0.len(),
                    column
                        .iter()
                        .flat_map(|icscb| icscb.eps_nu().iter().cloned().collect::<Vec<Float>>()),
                );

                let matrix_eps_s = DMatrix::from_iterator(
                    3,
                    data.0.len(),
                    column
                        .iter()
                        .flat_map(|icscb| icscb.eps_s().iter().cloned().collect::<Vec<Float>>()),
                );

                PyList::new(
                    py,
                    vec![
                        matrix_ics.transpose().to_pyarray(py),
                        matrix_eps_mu.transpose().to_pyarray(py),
                        matrix_eps_nu.transpose().to_pyarray(py),
                        matrix_eps_s.transpose().to_pyarray(py),
                    ],
                )?
                .into_any()
            }
            PyObsData::P3IMG(data) => {
                let column = data.0.output(key);

                PyList::new(
                    py,
                    Vec::from_iter(column.iter().map(|image| image.transpose().to_pyarray(py))),
                )?
                .into_any()
            }
            PyObsData::P3V1(data) => {
                let column = data.0.output(key);

                let matrix = DMatrix::from_iterator(
                    1,
                    data.0.len(),
                    column.iter().flat_map(|value| value.iter().cloned()),
                );

                matrix.transpose().to_pyarray(py).into_any()
            }
            PyObsData::P3V3(data) => {
                let column = data.0.output(key);

                let matrix = DMatrix::from_iterator(
                    3,
                    data.0.len(),
                    column.iter().flat_map(|icscb| icscb.iter().cloned()),
                );

                matrix.transpose().to_pyarray(py).into_any()
            }
            PyObsData::P3V4(data) => {
                let column = data.0.output(key);

                let matrix = DMatrix::from_iterator(
                    4,
                    data.0.len(),
                    column.iter().flat_map(|icscb| icscb.iter().cloned()),
                );

                matrix.transpose().to_pyarray(py).into_any()
            }
        })
    }

    #[new]
    #[pyo3(signature = (obs, size, dtype, opt_ref_data = None))]
    /// Create a new PyObsData object.
    pub fn new(
        obs: &PyObs,
        size: usize,
        dtype: PyObsDataType,
        opt_ref_data: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        Ok(match opt_ref_data {
            Some(ref_data) => match dtype {
                PyObsDataType::B3 => {
                    return Err(PyValueError::new_err(
                        "ICSBasis data with reference values are not supported",
                    ));
                }
                PyObsDataType::P3IMG => {
                    let list = py_any_iterator!(ref_data, PyReadonlyArray2<Float>)
                        .map(|value| {
                            Ok(ObsImg(array_to_matrix::<
                                Dim<[usize; 2]>,
                                Dyn,
                                Dyn,
                                Dyn,
                                Dyn,
                            >(value)?))
                        })
                        .collect::<Result<Vec<ObsImg<Float>>, PyErr>>()?;

                    PyObsData::P3IMG(PyObsP3IMG(py_unwrap!(
                        ObsEnsbl::new(
                            obs.as_cam()?.clone(),
                            size,
                            Some(DVector::from(Vec::from_iter(list))),
                        ),
                        "Observation must have same length as the reference data"
                    )))
                }
                PyObsDataType::P3V1 => {
                    let iter = py_any_iterator!(ref_data, Float).map(|value| ObsVec::from([value]));

                    PyObsData::P3V1(PyObsP3V1(py_unwrap!(
                        ObsEnsbl::new(
                            obs.as_obs3()?.clone(),
                            size,
                            Some(DVector::from(Vec::from_iter(iter))),
                        ),
                        "Observation must have same length as the reference data"
                    )))
                }
                PyObsDataType::P3V3 => {
                    let iter = py_any_iterator!(ref_data, [Float; 3]).map(ObsVec::from);

                    PyObsData::P3V3(PyObsP3V3(py_unwrap!(
                        ObsEnsbl::new(
                            obs.as_obs3()?.clone(),
                            size,
                            Some(DVector::from(Vec::from_iter(iter))),
                        ),
                        "Observation must have same length as the reference data"
                    )))
                }
                PyObsDataType::P3V4 => {
                    let iter = py_any_iterator!(ref_data, [Float; 4]).map(ObsVec::from);

                    PyObsData::P3V4(PyObsP3V4(py_unwrap!(
                        ObsEnsbl::new(
                            obs.as_obs3()?.clone(),
                            size,
                            Some(DVector::from(Vec::from_iter(iter))),
                        ),
                        "Observation must have same length as the reference data"
                    )))
                }
            },
            None => match dtype {
                PyObsDataType::B3 => PyObsData::B3(PyICSBasis3(
                    ObsEnsbl::new(obs.as_obs3()?.clone(), size, None).unwrap(),
                )),
                PyObsDataType::P3IMG => PyObsData::P3IMG(PyObsP3IMG(
                    ObsEnsbl::new(obs.as_cam()?.clone(), size, None).unwrap(),
                )),
                PyObsDataType::P3V1 => PyObsData::P3V1(PyObsP3V1(
                    ObsEnsbl::new(obs.as_obs3()?.clone(), size, None).unwrap(),
                )),
                PyObsDataType::P3V3 => PyObsData::P3V3(PyObsP3V3(
                    ObsEnsbl::new(obs.as_obs3()?.clone(), size, None).unwrap(),
                )),
                PyObsDataType::P3V4 => PyObsData::P3V4(PyObsP3V4(
                    ObsEnsbl::new(obs.as_obs3()?.clone(), size, None).unwrap(),
                )),
            },
        })
    }

    /// Return the size of the observation data ensemble.
    pub fn size(&self) -> usize {
        match self {
            PyObsData::B3(data) => data.0.len(),
            PyObsData::P3IMG(data) => data.0.len(),
            PyObsData::P3V1(data) => data.0.len(),
            PyObsData::P3V3(data) => data.0.len(),
            PyObsData::P3V4(data) => data.0.len(),
        }
    }
}
