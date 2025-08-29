use crate::{
    util::array_to_matrix, Float, PyDiagObser, PyImageObser, PyMagScObs, PyObser, PyUnivariate,
};
use nalgebra::{Const, DMatrix, Dyn, SVector, U1};
use numpy::{ndarray::Dim, PyArray2, PyReadonlyArray2, ToPyArray};
use ocnus::{
    base::{Model, ModelEnsbl, Obser},
    coords::Coordinates,
    models,
    obsty::{ICSCoordsBasis, InSituPlasmaDensity, NullNoise, RemoteWhiteLight},
    stats::MultivariateDensity,
};
use paste::paste;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyType};

/// Implement python types for single-observable vector models.
macro_rules! impl_vector_model {
    ($name: literal, $obsname: literal, $model: ty, $dim: expr, $csst: ty, $fmst: ty, $simulate: expr, $simulate_type: ty) => {
        paste! {
            #[allow(missing_docs)]
            #[derive(Clone)]
            #[pyclass]
            pub struct [<$name Ensbl>](pub ModelEnsbl<Float, $model<Float, MultivariateDensity<Float, $dim>>, $dim>);
        }

        paste! {
            #[pymethods]
            impl [<$name Ensbl>] {
                #[allow(missing_docs)]
                pub fn __getitem__<'py>(&self, py: Python<'py>, key: usize) -> Bound<'py, PyArray2<Float>> {
                    let column = self.0.ptpdf.get_particle(key);

                    let matrix = DMatrix::from_iterator(
                        $dim,
                        1,
                        column.iter().copied(),
                    );

                    matrix.transpose().to_pyarray(py)
                }

                #[allow(missing_docs)]
                #[classmethod]
                pub fn from_particles(_cls: &Bound<PyType>, array: PyReadonlyArray2<Float>) -> PyResult<Self> {
                    Ok(Self(ModelEnsbl::from_particles(array_to_matrix::<Dim<[usize; 2]>, Dyn, Const<$dim>, U1, Dyn>(array)?, None, None)))
                }

                #[allow(missing_docs)]
                #[new]
                pub fn new(size: usize) -> Self {
                    Self(ModelEnsbl::new(size, None))
                }

                #[allow(missing_docs)]
                pub fn set_particle(&mut self, index: usize, values: &Bound<PyAny>) -> PyResult<()> {
                    let iter = match values.try_iter() {
                        Ok(value) => value.map(|py_obj| py_obj.unwrap().extract::<Float>().unwrap()),
                        Err(..) => return Err(PyValueError::new_err("values argument must be iterable")),
                    };

                    let vector = SVector::<Float, $dim>::from_iterator(iter);

                    self.0.ptpdf.set_particle(index, &vector.as_view());

                    Ok(())
                }
            }
        }

        paste! {
            #[allow(missing_docs)]
            #[pyclass]
            pub struct [<$name Model>](pub $model<Float, MultivariateDensity<Float, $dim>>);}

            paste! {
            #[pymethods]
            impl [<$name Model>] {
                #[allow(missing_docs)]
                pub fn diagnostics<'py>(&self, py: Python<'py>, scobs: &PyMagScObs, ensbl: &mut [<$name Ensbl>]) -> PyDiagObser {
                    py.allow_threads(|| {
                        self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                        let mut obser = Obser::<Float, ICSCoordsBasis<Float>>::new(scobs.0.as_scobs(), ensbl.0.len());

                        self.0.simulate_ics_basis_ensbl(&mut ensbl.0, &mut obser).unwrap();

                        PyDiagObser(obser)
                    })
                }

                #[allow(missing_docs)]
                #[new]
                pub fn new(priors: Vec<PyUnivariate>) -> PyResult<Self> {
                    let names = $model::<Float, MultivariateDensity<Float, $dim>>::PARAMS;

                    if priors.len() != $dim {
                        Err(PyValueError::new_err("invalid number of parameters"))
                    } else {
                        if priors.iter().zip(names.iter()).fold(true, |acc, next| {
                            acc & (next.0.0.0 == *next.1)
                        }) {
                            let mvpdf = MultivariateDensity::new(priors.iter().map(|uvpdf| &uvpdf.0 .1));

                            Ok(Self($model::<Float, MultivariateDensity<Float, $dim>>::new(mvpdf)))
                        } else {
                            Err(PyValueError::new_err("invalid parameter names"))
                        }
                    }
                }

                #[allow(missing_docs)]
                pub fn [<$obsname _simulate>]<'py>(&self, py: Python<'py>, ensbl: &mut [<$name Ensbl>], obser: &mut $simulate_type)  {
                    py.allow_threads(|| {
                        self.0.initialize_states_ensbl(&mut ensbl.0).unwrap();

                        self.0.simulate_ensbl(&mut ensbl.0, &mut obser.0, &$simulate,  &mut None::<&mut NullNoise<Float>>).unwrap();
                    })
                }
            }
        }
    };
}

impl_vector_model!(
    "COREWL",
    "wl",
    models::COREModel,
    11,
    coordinate::TTState<Float>,
    models::COREState<Float>,
    models::COREModel::observe_rwl,
    PyImageObser
);

impl_vector_model!(
    "CORERHO",
    "rho",
    models::COREModel,
    11,
    coordinate::TTState<Float>,
    models::COREState<Float>,
    models::COREModel::observe_np,
    PyObser
);
