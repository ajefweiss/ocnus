use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use nalgebra::{DMatrix, RealField, Scalar};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    ops::{Add, AddAssign},
};

use crate::obsty::Observable;

/// Generic observation image.
#[derive(
    Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, PartialEq, Serialize,
)]
#[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
pub struct ObserImg<T>(#[into_iterator(owned, ref, ref_mut)] pub DMatrix<T>)
where
    T: Clone + Scalar;

impl<T> Add for ObserImg<T>
where
    T: RealField,
{
    type Output = ObserImg<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(self.0.nrows() == rhs.nrows());

        Self(DMatrix::from_iterator(
            self.0.nrows(),
            self.0.ncols(),
            self.0
                .iter()
                .zip(rhs.iter())
                .map(|(a, b)| a.clone() + b.clone()),
        ))
    }
}

impl<T> AddAssign for ObserImg<T>
where
    T: RealField,
{
    fn add_assign(&mut self, rhs: Self) {
        assert!(self.0.nrows() == rhs.nrows());

        self.0
            .iter_mut()
            .zip(rhs.0.iter())
            .for_each(|(a, b)| *a += b.clone())
    }
}

impl<T> Default for ObserImg<T>
where
    T: Copy + RealField,
{
    fn default() -> Self {
        Self(DMatrix::zeros(0, 0))
    }
}

impl<T> Observable for ObserImg<T>
where
    T: Copy + RealField,
{
    fn is_valid(&self) -> bool {
        self.0.nrows() != 0
    }
}

impl<T> Zero for ObserImg<T>
where
    T: Copy + RealField,
{
    fn is_zero(&self) -> bool {
        self.0.nrows() == 0
    }

    fn set_zero(&mut self) {
        self.0 = DMatrix::zeros(0, 0)
    }

    fn zero() -> Self {
        Self(DMatrix::zeros(0, 0))
    }
}
