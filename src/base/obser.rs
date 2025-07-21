use crate::obsty::ICSCoordsBasis;
use derive_more::IntoIterator;
use log::debug;
use nalgebra::{
    DMatrix, DVectorView, Dyn, MatrixView, MatrixViewMut, RealField, Scalar, U1, Vector3,
};
use num_traits::Zero;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    ops::{Add, AddAssign},
};

/// A data structure holding spacecraft observations and an appropriately sized model output array.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Obser<T, OT>
where
    T: Copy + Scalar,
    OT: Clone + Scalar,
{
    /// Model output array.
    output: DMatrix<OT>,

    /// Spacecraft observations.
    scobs: ScObs<T, OT>,
}

impl<T, OT> Obser<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    OT: Scalar + Zero,
{
    /// Return a list of error values using a given error metric.
    pub fn errors<EF>(&self, func: &EF) -> Vec<T>
    where
        T: Send + Sync,
        OT: Send + Sync,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
    {
        self.par_ensbl_iter()
            .map(|(_, out)| func(self.scobs.refdt(), out.as_slice()))
            .collect::<Vec<T>>()
    }

    /// Return a list of error values and flags using a threshold value using a given error metric.
    pub fn errors_with_threshold<EF>(&self, func: &EF, threshold: T) -> (Vec<T>, Vec<bool>)
    where
        T: Send + Sync,
        OT: Send + Sync,
        EF: Fn(&[OT], &[OT]) -> T + Sync,
    {
        let mut flags = vec![true; self.output.ncols()];

        let values = self
            .par_ensbl_iter()
            .zip(flags.par_iter_mut())
            .chunks(128)
            .map(|mut chunks| {
                chunks
                    .iter_mut()
                    .map(|((_, out), flag)| {
                        let value = func(self.scobs.refdt(), out.as_slice());

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
    pub fn get_output(&self, index: usize) -> DVectorView<OT> {
        self.output.column(index)
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.scobs.len()
    }

    /// Create a new [`Obser`].
    pub fn new(scobs: ScObs<T, OT>, size: usize) -> Self {
        let slen = scobs.len();

        Self {
            scobs,
            output: DMatrix::<OT>::zeros(slen, size),
        }
    }

    /// Create a new [`Obser`] with custom observable type without copying the reference observations.
    pub fn new_as<OT2>(scobs: ScObs<T, OT>, size: usize) -> Obser<T, OT2>
    where
        OT2: Scalar + Zero,
    {
        let slen = scobs.len();

        Obser {
            scobs: ScObs::<T, OT2> {
                refdt: None,
                tconf: scobs.tconf,
                sorti: scobs.sorti,
            },
            output: DMatrix::<OT2>::zeros(slen, size),
        }
    }

    /// Iterate over the ensemble in parallel.
    pub fn par_ensbl_iter(
        &self,
    ) -> impl IndexedParallelIterator<Item = (&ScObs<T, OT>, MatrixView<OT, Dyn, U1>)>
    where
        T: Send + Sync,
        OT: Send + Sync,
    {
        self.output.par_column_iter().map(|col| (&self.scobs, col))
    }

    /// Mutably iterate over the ensemble in parallel.
    pub fn par_ensbl_iter_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = (&ScObs<T, OT>, MatrixViewMut<OT, Dyn, U1>)>
    where
        T: Send + Sync,
        OT: Send + Sync,
    {
        self.output
            .par_column_iter_mut()
            .map(|col| (&self.scobs, col))
    }

    /// Return a reference to the output array.
    pub fn output(&self) -> &DMatrix<OT> {
        &self.output
    }

    /// Returns the reference observations, of the internal [`ScObs`], as a slice.
    pub fn refdt(&self) -> &[OT] {
        self.scobs.refdt()
    }

    /// Return a reference to the internal [`ScObs`]
    pub fn scobs(&self) -> &ScObs<T, OT> {
        &self.scobs
    }

    /// Set an individual output column to the given value.
    pub fn set_output(&mut self, index: usize, column: &DVectorView<OT>) {
        self.output.set_column(index, column)
    }

    /// Return the size of the ensemble.
    pub fn size(&self) -> usize {
        self.output.ncols()
    }

    /// Iterate over the ensemble along the time axis.
    #[allow(clippy::type_complexity)]
    pub fn time_iter(
        &mut self,
    ) -> impl Iterator<Item = (T, &ScConf<T>, MatrixView<OT, U1, Dyn, U1, Dyn>)> {
        (&self.scobs)
            .into_iter()
            .zip(self.output.row_iter())
            .map(|(tconf, row)| (tconf.0, &tconf.1, row))
    }

    /// Mutably iterate over the ensemble along the time axis.
    #[allow(clippy::type_complexity)]
    pub fn time_iter_mut(
        &mut self,
    ) -> impl Iterator<Item = (T, &ScConf<T>, MatrixViewMut<OT, U1, Dyn, U1, Dyn>)> {
        (&self.scobs)
            .into_iter()
            .zip(self.output.row_iter_mut())
            .map(|(tconf, row)| (tconf.0, &tconf.1, row))
    }
}

/// The configuration of a single spacecraft observation, as used in [`ScObs`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ScConf<T>
where
    T: Copy + Scalar,
{
    /// Position in space, in an arbitrary Solar centric coordiante system.
    Position(Vector3<T>),
    /// Position in space, in an arbitrary Solar centric coordiante system,
    /// with 5 extra points defining a camera viewport (center, left, right, top, bottom).
    PositionViewport((Vector3<T>, [Vector3<T>; 5])),
}

impl<T> ScConf<T>
where
    T: Copy + Scalar,
{
    /// Computes distance between `self` and `other`.
    pub fn distance(&self, other: &Self) -> T
    where
        T: Copy + RealField,
    {
        (self.position() - other.position()).norm()
    }

    /// Returns the position of the spacecraft.
    pub fn position(&self) -> &Vector3<T> {
        match &self {
            ScConf::Position(r_self) => r_self,
            ScConf::PositionViewport(rvp_self) => &rvp_self.0,
        }
    }
}

/// A scobs of spacecraft observations with optional observation data.
#[derive(Clone, Debug, Default, Deserialize, IntoIterator, Serialize)]
#[serde(bound(serialize = "T: Serialize, OT: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>, OT: Deserialize<'de>"))]
pub struct ScObs<T, OT>
where
    T: Copy + Scalar,
    OT: Clone + Scalar,
{
    #[serde(skip)]
    /// Optional vector with reference observations.
    refdt: Option<Vec<OT>>,

    /// Vector of timestamps and spacecraft configurations.
    #[into_iterator(ref)]
    tconf: Vec<(T, ScConf<T>)>,

    /// The sorting indices that are used to recover the original [`ScObs`]
    /// objects from a composite scobs.
    sorti: Vec<usize>,
}

impl<T, OT> ScObs<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    OT: Clone + Scalar,
{
    /// Return a new [`ScObs`] with a different observable type.
    pub fn as_scobs<OT2>(&self) -> ScObs<T, OT2>
    where
        OT2: Clone + Scalar,
    {
        ScObs::<T, OT2>::from_iterator(self.clone())
    }

    /// Returns the number of individual [`ScObs`] contained within.
    pub fn count_series(&self) -> usize {
        self.sorti.iter().fold(0, |acc, next| max(acc, *next)) + 1
    }

    /// Returns a copy of itself for a diagnostic observable.
    pub fn diag(&self) -> ScObs<T, ICSCoordsBasis<T>>
    where
        T: RealField,
    {
        ScObs {
            refdt: None,
            tconf: self.tconf.clone(),
            sorti: self.sorti.clone(),
        }
    }

    /// Returns the first timestamp and [`ScConf`], if possible.
    pub fn first(&self) -> Option<&(T, ScConf<T>)> {
        self.tconf.first()
    }

    /// Create a [`ScObs`] from an iterator over `(T, ScConf<T>)`.
    pub fn from_iterator<I: IntoIterator<Item = (T, ScConf<T>)>>(iter: I) -> Self {
        let tconf = iter.into_iter().collect::<Vec<(T, ScConf<T>)>>();
        let length = tconf.len();

        Self {
            refdt: None,
            tconf,
            sorti: vec![0; length],
        }
    }

    /// Create a [`ScObs`] from an iterator over `(T, ScConf<T>, OT)`.
    pub fn from_iterator_with_observables<I: IntoIterator<Item = (T, ScConf<T>, OT)>>(
        iter: I,
    ) -> Self {
        let (tconf, refdt): (Vec<(T, ScConf<T>)>, Vec<OT>) = iter
            .into_iter()
            .map(|value| ((value.0, value.1), value.2))
            .unzip();

        let length = tconf.len();

        Self {
            refdt: Some(refdt),
            tconf,
            sorti: vec![0; length],
        }
    }

    /// Returns `true`` if the observation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.tconf.is_empty()
    }

    /// Returns the last timestamp and [`ScConf`], if possible.
    pub fn last_scobs(&self) -> Option<&(T, ScConf<T>)> {
        self.tconf.last()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.tconf.len()
    }

    /// Create an empty [`ScObs`].
    pub fn new() -> Self {
        Self {
            refdt: None,
            tconf: Vec::new(),
            sorti: Vec::new(),
        }
    }

    /// Returns the reference observations as a slice.
    pub fn refdt(&self) -> &[OT] {
        self.refdt
            .as_ref()
            .expect("no reference observations")
            .as_slice()
    }

    /// Return a list of timestamps.
    pub fn timestamps(&self) -> Vec<T> {
        self.tconf
            .iter()
            .map(|(timestamp, _)| *timestamp)
            .collect::<Vec<T>>()
    }

    /// Sorts the underlying `Vec<ScObs>` object by the time stamp field.
    pub fn sort_by_timestamp(&mut self) {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort_closure = |refdt: &mut Option<Vec<OT>>,
                                   tconf: &mut Vec<(T, ScConf<T>)>,
                                   sorti: &mut Vec<usize>| {
            let mut counter = 0;

            for idx in 0..(tconf.len() - 1) {
                if tconf[idx].0 > tconf[idx + 1].0 {
                    if refdt.is_some() {
                        refdt.as_mut().unwrap().swap(idx, idx + 1);
                    }

                    tconf.swap(idx, idx + 1);
                    sorti.swap(idx, idx + 1);

                    counter += 1
                }
            }

            counter
        };

        let mut counter = 1;

        while counter != 0 {
            counter = bubble_sort_closure(&mut self.refdt, &mut self.tconf, &mut self.sorti);
        }
    }
}

impl<T, OT> Add for ScObs<T, OT>
where
    T: Copy + Scalar,
    OT: Clone + Scalar,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut refdt = self.refdt;
        let mut tconf = self.tconf;

        debug!(
            "merging two ScObs objects ({} + {})",
            tconf.len(),
            rhs.tconf.len()
        );

        tconf.extend(rhs.tconf);

        if let Some(refdt_uw) = &mut refdt {
            refdt_uw.extend(rhs.refdt.unwrap());
        }

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |acc: usize, &v| max(acc, v)) + 1;

        let mut sorti = self.sorti;

        // Add index_offset to all indices in rhs.
        sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        Self {
            refdt,
            tconf,
            sorti,
        }
    }
}

impl<T, OT> AddAssign for ScObs<T, OT>
where
    T: Copy + Scalar,
    OT: Clone + Scalar,
{
    fn add_assign(&mut self, rhs: Self) {
        debug!(
            "merging two ScObs objects ({} + {})",
            self.tconf.len(),
            rhs.tconf.len()
        );

        if self.refdt.is_some() {
            self.refdt.as_mut().unwrap().extend(rhs.refdt.unwrap());
        }

        self.tconf.extend(rhs.tconf);

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |acc, &v| max(acc, v)) + 1;

        // Add index_offset to all indices in rhs.
        self.sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );
    }
}

impl<T, I1, I2, OT> From<(I1, I2)> for ScObs<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = ScConf<T>>,
    OT: Clone + Scalar,
{
    fn from(value: (I1, I2)) -> Self {
        Self::from_iterator(value.0.into_iter().zip(value.1))
    }
}

impl<T, I, OT> From<(I, ScConf<T>)> for ScObs<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    I: IntoIterator<Item = T>,
    OT: Clone + Scalar,
{
    fn from(value: (I, ScConf<T>)) -> Self {
        Self::from_iterator(
            value
                .0
                .into_iter()
                .map(|timestamp| (timestamp, value.1.clone())),
        )
    }
}

impl<T, I1, I2, OT> From<(I1, ScConf<T>, I2)> for ScObs<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = OT>,
    OT: Clone + Scalar,
{
    fn from(value: (I1, ScConf<T>, I2)) -> Self {
        Self::from_iterator_with_observables(
            value
                .0
                .into_iter()
                .zip(value.2)
                .map(|(timestamp, observable)| (timestamp, value.1.clone(), observable.clone())),
        )
    }
}

impl<T, I1, I2, I3, OT> From<(I1, I2, I3)> for ScObs<T, OT>
where
    T: Copy + PartialOrd + Scalar,
    I1: IntoIterator<Item = T>,
    I2: IntoIterator<Item = ScConf<T>>,
    I3: IntoIterator<Item = OT>,
    OT: Clone + Scalar,
{
    fn from(value: (I1, I2, I3)) -> Self {
        Self::from_iterator_with_observables(
            value
                .0
                .into_iter()
                .zip(value.1)
                .zip(value.2)
                .map(|((timestamp, conf), observable)| {
                    (timestamp, conf.clone(), observable.clone())
                }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scobs() {
        let sctc1: ScObs<f64, f64> =
            ScObs::from((vec![0.0], ScConf::Position(Vector3::new(1.0, 0.0, 0.0))));
        let sctc2 = ScObs::from((vec![1.0], ScConf::Position(Vector3::new(1.0, 0.0, 0.0))));
        let sctc3 = ScObs::from((vec![0.5], ScConf::Position(Vector3::new(1.0, 0.0, 0.0))));

        let sctc12 = sctc1 + sctc2;

        assert!(sctc12.count_series() == 2);
        assert!(sctc3.count_series() == 1);
        assert!(sctc3.last_scobs().unwrap().0 == 0.5);

        let mut sctc123 = sctc12 + sctc3;

        assert!(sctc123.count_series() == 3);
        assert!(sctc123.len() == 3);
        assert!(!sctc123.is_empty());

        sctc123.sort_by_timestamp();

        assert!(sctc123.tconf.first().unwrap().0 == 0.0);
        assert!(sctc123.tconf.last().unwrap().0 == 1.0);
    }
}
