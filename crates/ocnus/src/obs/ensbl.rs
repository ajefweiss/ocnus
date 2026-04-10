use nalgebra::{DMatrix, DVector, DVectorView, Dyn, MatrixView, MatrixViewMut, Scalar, U1};
use num_traits::Zero;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::obs::{Obs, conf::ObsTime};

/// A data structure that holds an observation, with optional reference
/// data, and an appropriately sized output array for the an ensemble model.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ObsEnsbl<T, OC, OD>
where
    T: Scalar,
    OC: Scalar,
    OD: Scalar,
{
    /// Observer configuration time-series.
    obs: Obs<T, OC>,

    /// Observable ensemble array.
    outputs: DMatrix<OD>,

    /// Observable reference array (optional).
    ref_data: Option<DVector<OD>>,
}

impl<T, OC, OD> ObsEnsbl<T, OC, OD>
where
    T: Copy + PartialOrd + Scalar,
    OC: ObsTime<T>,
    OD: Scalar + Zero,
{
    /// Iterate over the ensemble.
    pub fn ensbl_iter(&self) -> impl Iterator<Item = (&Obs<T, OC>, MatrixView<'_, OD, Dyn, U1>)> {
        self.outputs.column_iter().map(|col| (&self.obs, col))
    }

    /// Mutably iterate over the ensemble.
    pub fn ensbl_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (&Obs<T, OC>, MatrixViewMut<'_, OD, Dyn, U1>)> {
        self.outputs.column_iter_mut().map(|col| (&self.obs, col))
    }

    /// Return a list of error values for a given error metric.
    pub fn errors_func<EF>(&self, func: &EF) -> Vec<T>
    where
        T: Send + Sync,
        OC: Sync,
        OD: Send + Sync,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        self.par_ensbl_iter()
            .map(|(_, out)| {
                func(
                    self.ref_data
                        .as_ref()
                        .expect("reference data missing")
                        .as_slice(),
                    out.as_slice(),
                )
            })
            .collect::<Vec<T>>()
    }

    /// Return a list of error values and flags using a threshold value for a given error metric.
    pub fn errors_with_threshold<EF>(&self, func: &EF, threshold: T) -> (Vec<T>, Vec<bool>)
    where
        T: Send + Sync,
        OC: Sync,
        OD: Send + Sync,
        EF: Fn(&[OD], &[OD]) -> T + Sync,
    {
        let mut flags = vec![true; self.outputs.ncols()];

        let values = self
            .par_ensbl_iter()
            .zip(flags.par_iter_mut())
            .chunks(128)
            .map(|mut chunk| {
                chunk
                    .iter_mut()
                    .map(|((_, out), flag)| {
                        let value = func(
                            self.ref_data
                                .as_ref()
                                .expect("reference data missing")
                                .as_slice(),
                            out.as_slice(),
                        );

                        **flag = value < threshold;

                        value
                    })
                    .collect::<Vec<T>>()
            })
            .flatten()
            .collect::<Vec<T>>();

        (values, flags)
    }

    /// Return a reference to an individual output column.
    pub fn output(&self, index: usize) -> DVectorView<'_, OD> {
        assert!(
            index < self.outputs.ncols(),
            "cannot get output, index out of bounds"
        );

        self.outputs.column(index)
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.obs.len()
    }

    /// Create a new [`crate::obs::data::ObsData`].
    pub fn new(obs: Obs<T, OC>, size: usize, opt_ref_data: Option<DVector<OD>>) -> Option<Self> {
        let slen = obs.len();

        match opt_ref_data.as_ref() {
            Some(data) if data.len() != slen => None,
            _ => Some(Self {
                outputs: DMatrix::<OD>::zeros(slen, size),
                ref_data: opt_ref_data,
                obs,
            }),
        }
    }

    /// Iterate over the ensemble in parallel.
    pub fn par_ensbl_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = (&Obs<T, OC>, MatrixView<'_, OD, Dyn, U1>)>
    where
        T: Send + Sync,
        OC: Sync,
        OD: Send + Sync,
    {
        self.outputs.par_column_iter().map(|col| (&self.obs, col))
    }

    /// Mutably iterate over the ensemble in parallel.
    pub fn par_ensbl_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = (&Obs<T, OC>, MatrixViewMut<'_, OD, Dyn, U1>)>
    where
        T: Send + Sync,
        OC: Sync,
        OD: Send + Sync,
    {
        self.outputs
            .par_column_iter_mut()
            .map(|col| (&self.obs, col))
    }

    /// Returns the reference observations, of the internal [`Obs`], as a slice.
    pub fn ref_data(&self) -> Option<&DVector<OD>> {
        self.ref_data.as_ref()
    }

    /// Return a reference to the internal [`Obs`]
    pub fn obs(&self) -> &Obs<T, OC> {
        &self.obs
    }

    /// Set an individual output column to the given value.
    pub fn set_output(&mut self, index: usize, column: &DVectorView<OD>) {
        assert!(
            index < self.outputs.ncols(),
            "cannot set output, index out of bounds"
        );

        self.outputs.set_column(index, column)
    }

    /// Return the size of the ensemble.
    pub fn size(&self) -> usize {
        self.outputs.ncols()
    }

    /// Iterate over the ensemble along the time axis.
    pub fn time_iter(
        &mut self,
    ) -> impl Iterator<Item = (&OC, MatrixView<'_, OD, U1, Dyn, U1, Dyn>)> {
        (&self.obs).into_iter().zip(self.outputs.row_iter())
    }

    /// Mutably iterate over the ensemble along the time axis.
    pub fn time_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (&OC, MatrixViewMut<'_, OD, U1, Dyn, U1, Dyn>)> {
        (&self.obs).into_iter().zip(self.outputs.row_iter_mut())
    }
}
