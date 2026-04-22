use crate::{
    Float,
    util::{array_to_matrix, py_any_iterator},
};
use nalgebra::{Dyn, OMatrix, SVector, U1, U2, U3, Vector1, Vector2, Vector3};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray, ndarray::Dim};
use ocnus::obs::{
    Obs,
    conf::{CamConf, ObsPosition, VecConf},
};
use paste::paste;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};
use wcs::WCSParams;

#[allow(missing_docs)]
#[pyclass(from_py_object, name = "Obs")]
#[derive(Clone)]
pub enum PyObs {
    Obs0(PyObs0),
    Obs1(PyObs1),
    Obs2(PyObs2),
    Obs3(PyObs3),
    ObsCam(PyObsCam),
}

#[allow(missing_docs)]
#[pyclass(from_py_object, name = "ObsType")]
#[derive(Clone)]
pub enum PyObsType {
    Obs0,
    Obs1,
    Obs2,
    Obs3,
    ObsCam,
}

macro_rules! impl_pyobs {
    ($ndim: expr, $name: literal) => {
        paste! {

            #[pyclass(from_py_object, name = $name)]
            #[derive(Clone)]
            #[doc="PyObs for n="  $ndim]
            pub struct [<PyObs $ndim>](pub Obs<Float, VecConf<Float, $ndim>>);
        }
    };
}

impl_pyobs!(0, "Obs0");
impl_pyobs!(1, "Obs1");
impl_pyobs!(2, "Obs2");
impl_pyobs!(3, "Obs3");

#[pyclass(from_py_object, name = "ObsCam")]
#[derive(Clone)]
/// PyObs for camera observations.
pub struct PyObsCam(pub Obs<Float, CamConf<Float>>);

impl PyObs {
    /// Return inner Obs with VecConf<T, 0>, or error if not of that type.
    pub fn as_obs0(&self) -> PyResult<&Obs<Float, VecConf<Float, 0>>> {
        match self {
            PyObs::Obs0(obs) => Ok(&obs.0),
            _ => Err(PyValueError::new_err("Observation is not of type Obs0")),
        }
    }

    /// Return inner Obs with VecConf<T, 1>, or error if not of that type.
    pub fn as_obs1(&self) -> PyResult<&Obs<Float, VecConf<Float, 1>>> {
        match self {
            PyObs::Obs1(obs) => Ok(&obs.0),
            _ => Err(PyValueError::new_err("Observation is not of type Obs1")),
        }
    }

    /// Return inner Obs with VecConf<T, 2>, or error if not of that type.
    pub fn as_obs2(&self) -> PyResult<&Obs<Float, VecConf<Float, 2>>> {
        match self {
            PyObs::Obs2(obs) => Ok(&obs.0),
            _ => Err(PyValueError::new_err("Observation is not of type Obs2")),
        }
    }

    /// Return inner Obs with VecConf<T, 3>, or error if not of that type.
    pub fn as_obs3(&self) -> PyResult<&Obs<Float, VecConf<Float, 3>>> {
        match self {
            PyObs::Obs3(obs) => Ok(&obs.0),
            _ => Err(PyValueError::new_err("Observation is not of type Obs3")),
        }
    }

    /// Return inner Obs with CamConf, or error if not of that type.
    pub fn as_cam(&self) -> PyResult<&Obs<Float, CamConf<Float>>> {
        match self {
            PyObs::ObsCam(obs) => Ok(&obs.0),
            _ => Err(PyValueError::new_err("Observation is not of type ObsCam")),
        }
    }
}

#[pymethods]
impl PyObs {
    /// Combine two observation sets of the same type into one.
    #[classmethod]
    pub fn combine(_cls: &Bound<PyType>, obs_a: &PyObs, obs_b: &PyObs) -> PyResult<Self> {
        match (obs_a, obs_b) {
            (PyObs::Obs0(a), PyObs::Obs0(b)) => {
                let mut obs = a.0.clone() + b.0.clone();
                obs.sort_by_timestamp();
                Ok(PyObs::Obs0(PyObs0(obs)))
            }
            (PyObs::Obs1(a), PyObs::Obs1(b)) => {
                let mut obs = a.0.clone() + b.0.clone();
                obs.sort_by_timestamp();
                Ok(PyObs::Obs1(PyObs1(obs)))
            }
            (PyObs::Obs2(a), PyObs::Obs2(b)) => {
                let mut obs = a.0.clone() + b.0.clone();
                obs.sort_by_timestamp();
                Ok(PyObs::Obs2(PyObs2(obs)))
            }
            (PyObs::Obs3(a), PyObs::Obs3(b)) => {
                let mut obs = a.0.clone() + b.0.clone();
                obs.sort_by_timestamp();
                Ok(PyObs::Obs3(PyObs3(obs)))
            }
            (PyObs::ObsCam(a), PyObs::ObsCam(b)) => {
                let mut obs = a.0.clone() + b.0.clone();
                obs.sort_by_timestamp();
                Ok(PyObs::ObsCam(PyObsCam(obs)))
            }
            _ => Err(PyValueError::new_err(
                "Cannot combine observations of different types",
            )),
        }
    }

    /// Return the uncombined indices for the observations.
    pub fn uncombined_indices(&self) -> PyResult<Vec<usize>> {
        match self {
            PyObs::Obs0(obs) => Ok(obs.0.uncombined_indices()),
            PyObs::Obs1(obs) => Ok(obs.0.uncombined_indices()),
            PyObs::Obs2(obs) => Ok(obs.0.uncombined_indices()),
            PyObs::Obs3(obs) => Ok(obs.0.uncombined_indices()),
            PyObs::ObsCam(obs) => Ok(obs.0.uncombined_indices()),
        }
    }

    /// Return the number of observations.
    pub fn count(&self) -> usize {
        match self {
            PyObs::Obs0(obs) => obs.0.len(),
            PyObs::Obs1(obs) => obs.0.len(),
            PyObs::Obs2(obs) => obs.0.len(),
            PyObs::Obs3(obs) => obs.0.len(),
            PyObs::ObsCam(obs) => obs.0.len(),
        }
    }

    /// Return the camera field of views as a list of numpy arrays.
    fn camera_fovs<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        match self {
            PyObs::ObsCam(obs) => Ok(obs
                .0
                .configurations()
                .iter()
                .map(|conf| conf.fovs().transpose().to_pyarray(py).into_any())
                .collect()),
            _ => Err(PyValueError::new_err("Observation is not of type ObsCam")),
        }
    }

    #[new]
    #[pyo3(signature = (timestamps, opt_position = None, opt_wcs_string = None, opt_wsc_pol = None))]
    /// Construct a new PyObs instance.
    pub fn new(
        timestamps: Vec<Float>,
        opt_position: Option<PyReadonlyArray2<Float>>,
        opt_wcs_string: Option<&Bound<PyAny>>,
        opt_wsc_pol: Option<&Bound<PyAny>>,
    ) -> PyResult<Self> {
        match opt_wcs_string {
            Some(wcs_string) => {
                // Convert position(s) into appropriate data type.
                let iter_wcs_string = py_any_iterator!(wcs_string, String);

                let opt_iter_wcs_pol = match opt_wsc_pol {
                    Some(wcs_pol) => Some(py_any_iterator!(wcs_pol, Float)),
                    None => None,
                };

                let array = match opt_position {
                    Some(position) => {
                        array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(position)?
                    }
                    None => {
                        return Err(PyValueError::new_err(
                            "Position array must be provided for camera observations",
                        ));
                    }
                };

                let ndim = array.nrows();

                if timestamps.len() != array.ncols() {
                    return Err(PyValueError::new_err(
                        "Number of timestamps must match number of position columns",
                    ));
                }

                if ndim != 3 {
                    return Err(PyValueError::new_err(
                        "Position array must have shape (3, N) for camera observations",
                    ));
                }

                match opt_iter_wcs_pol {
                    Some(iter_wcs_pol) => Ok(PyObs::ObsCam(PyObsCam(Obs::from_iter(
                        timestamps
                            .iter()
                            .zip(array.column_iter())
                            .zip(iter_wcs_string)
                            .zip(iter_wcs_pol)
                            .map(|(((ts, pos), wcs), wcs_pol)| {
                                let wcs_params: WCSParams =
                                    serde_json5::from_str::<WCSParams>(wcs.as_str())
                                        .expect("deserialization of wcs params failed");

                                CamConf::new(
                                    *ts,
                                    Vector3::from([pos[(0, 0)], pos[(1, 0)], pos[(2, 0)]]),
                                    wcs_params,
                                    Some(wcs_pol),
                                )
                            }),
                    )))),
                    None => Ok(PyObs::ObsCam(PyObsCam(Obs::from_iter(
                        timestamps
                            .iter()
                            .zip(array.column_iter())
                            .zip(iter_wcs_string)
                            .map(|((ts, pos), wcs)| {
                                let wcs_params: WCSParams =
                                    serde_json5::from_str::<WCSParams>(wcs.as_str())
                                        .expect("deserialization of wcs params failed");

                                CamConf::new(
                                    *ts,
                                    Vector3::from([pos[(0, 0)], pos[(1, 0)], pos[(2, 0)]]),
                                    wcs_params,
                                    None,
                                )
                            }),
                    )))),
                }
            }
            None => match opt_position {
                Some(position) => {
                    let array = array_to_matrix::<Dim<[usize; 2]>, Dyn, Dyn, Dyn, Dyn>(position)?;

                    let ndim = array.nrows();

                    if timestamps.len() != array.ncols() {
                        return Err(PyValueError::new_err(
                            "Number of timestamps must match number of position columns",
                        ));
                    }

                    match ndim {
                        1 => Ok(PyObs::Obs1(PyObs1(Obs::from_iter(
                            timestamps
                                .iter()
                                .zip(array.column_iter())
                                .map(|(ts, pos)| VecConf::new(*ts, Vector1::from([pos[(0, 0)]]))),
                        )))),
                        2 => Ok(PyObs::Obs2(PyObs2(Obs::from_iter(
                            timestamps.iter().zip(array.column_iter()).map(|(ts, pos)| {
                                VecConf::new(*ts, Vector2::from([pos[(0, 0)], pos[(1, 0)]]))
                            }),
                        )))),
                        3 => Ok(PyObs::Obs3(PyObs3(Obs::from_iter(
                            timestamps.iter().zip(array.column_iter()).map(|(ts, pos)| {
                                VecConf::new(
                                    *ts,
                                    Vector3::from([pos[(0, 0)], pos[(1, 0)], pos[(2, 0)]]),
                                )
                            }),
                        )))),
                        _ => Err(PyValueError::new_err(
                            "Position array must have shape (3, N), with N > 0 or be omitted",
                        )),
                    }
                }
                None => {
                    Ok(PyObs::Obs0(PyObs0(Obs::from_iter(timestamps.iter().map(
                        |ts| VecConf::new(*ts, SVector::<Float, 0>::zeros()),
                    )))))
                }
            },
        }
    }

    #[getter]
    /// Get the observation type as `PyObsType`
    pub fn obstype(&self) -> PyObsType {
        match self {
            PyObs::Obs0(_) => PyObsType::Obs0,
            PyObs::Obs1(_) => PyObsType::Obs1,
            PyObs::Obs2(_) => PyObsType::Obs2,
            PyObs::Obs3(_) => PyObsType::Obs3,
            PyObs::ObsCam(_) => PyObsType::ObsCam,
        }
    }

    /// Return positions as a 2D numpy array of shape (N_obs, ndim), or None if ndim=0.
    pub fn positions<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<Float>>> {
        match self {
            PyObs::Obs0(_) => None,
            PyObs::Obs1(obs) => Some(
                OMatrix::<Float, U1, Dyn>::from_columns(
                    obs.0
                        .configurations()
                        .iter()
                        .map(|conf| SVector::from(conf.position()))
                        .collect::<Vec<SVector<Float, 1>>>()
                        .as_slice(),
                )
                .transpose()
                .to_pyarray(py),
            ),
            PyObs::Obs2(obs) => Some(
                OMatrix::<Float, U2, Dyn>::from_columns(
                    obs.0
                        .configurations()
                        .iter()
                        .map(|conf| SVector::from(conf.position()))
                        .collect::<Vec<SVector<Float, 2>>>()
                        .as_slice(),
                )
                .transpose()
                .to_pyarray(py),
            ),
            PyObs::Obs3(obs) => Some(
                OMatrix::<Float, U3, Dyn>::from_columns(
                    obs.0
                        .configurations()
                        .iter()
                        .map(|conf| SVector::from(conf.position()))
                        .collect::<Vec<SVector<Float, 3>>>()
                        .as_slice(),
                )
                .transpose()
                .to_pyarray(py),
            ),
            PyObs::ObsCam(obs) => Some(
                OMatrix::<Float, U3, Dyn>::from_columns(
                    obs.0
                        .configurations()
                        .iter()
                        .map(|conf| SVector::from(conf.position()))
                        .collect::<Vec<SVector<Float, 3>>>()
                        .as_slice(),
                )
                .transpose()
                .to_pyarray(py),
            ),
        }
    }

    /// Return a new observation sorted by timestamp.
    pub fn sort_by_timestamp(&self) -> PyResult<Self> {
        let mut cloned = self.clone();

        match &mut cloned {
            PyObs::Obs0(obs) => obs.0.sort_by_timestamp(),
            PyObs::Obs1(obs) => obs.0.sort_by_timestamp(),
            PyObs::Obs2(obs) => obs.0.sort_by_timestamp(),
            PyObs::Obs3(obs) => obs.0.sort_by_timestamp(),
            PyObs::ObsCam(obs) => obs.0.sort_by_timestamp(),
        }

        Ok(cloned)
    }

    /// Return a subset of the observations given by the provided indices.
    pub fn subset(&self, indices: Vec<usize>) -> PyResult<Self> {
        match self {
            PyObs::Obs0(obs) => Ok(PyObs::Obs0(PyObs0(obs.0.subset(&indices)))),
            PyObs::Obs1(obs) => Ok(PyObs::Obs1(PyObs1(obs.0.subset(&indices)))),
            PyObs::Obs2(obs) => Ok(PyObs::Obs2(PyObs2(obs.0.subset(&indices)))),
            PyObs::Obs3(obs) => Ok(PyObs::Obs3(PyObs3(obs.0.subset(&indices)))),
            PyObs::ObsCam(obs) => Ok(PyObs::ObsCam(PyObsCam(obs.0.subset(&indices)))),
        }
    }

    /// Return the observation timestamps as a vector.
    pub fn timestamps(&self) -> Vec<Float> {
        match self {
            PyObs::Obs0(obs) => obs.0.timestamps(),
            PyObs::Obs1(obs) => obs.0.timestamps(),
            PyObs::Obs2(obs) => obs.0.timestamps(),
            PyObs::Obs3(obs) => obs.0.timestamps(),
            PyObs::ObsCam(obs) => obs.0.timestamps(),
        }
    }
}
