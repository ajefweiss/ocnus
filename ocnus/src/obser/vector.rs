//! Implementation of a `N`-dimensional vector observable.

use crate::obser::OcnusObser;
use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::SVector;
use num_traits::{Float, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

/// Generic N-dimensional observation vector.
#[derive(Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, Serialize)]
pub struct ObserVec<const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    pub [f32; N],
);

impl<const N: usize> ObserVec<N> {
    /// Returns true if any entry within the observation vector is `NaN`.
    pub fn any_nan(&self) -> bool {
        !self.iter().fold(true, |acc, next| acc & next.is_finite())
    }

    /// Returns true if all entries within the observation vector are `NaN`.
    pub fn is_nan(&self) -> bool {
        self.iter().fold(true, |acc, next| acc & next.is_nan())
    }

    /// Calculate the mean square error betweeon two observation vectors.
    pub fn mse(&self, other: &Self) -> f32 {
        (self - other).ss() / N as f32
    }

    /// Calculate the sum of squares over the entries within the observation vector.
    pub fn ss(&self) -> f32 {
        self.iter().map(|value| value.powi(2)).sum::<f32>()
    }

    /// Return a new observation vector filled with zeros.
    pub fn zeros() -> Self {
        ObserVec([0.0; N])
    }
}

impl<const N: usize> Add for ObserVec<N> {
    type Output = ObserVec<N>;

    fn add(self, rhs: Self) -> Self::Output {
        ObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Add<&'a ObserVec<N>> for &'a ObserVec<N> {
    type Output = ObserVec<N>;

    fn add(self, rhs: &'a ObserVec<N>) -> Self::Output {
        ObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 + v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> AddAssign for ObserVec<N> {
    fn add_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += rhs);
    }
}

impl<const N: usize> Default for ObserVec<N> {
    fn default() -> Self {
        ObserVec([f32::nan(); N])
    }
}

impl Display for ObserVec<3> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl Display for ObserVec<4> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} | {:.2}]",
            self[0], self[1], self[2], self[3]
        )
    }
}

impl<const N: usize> From<SVector<f32, N>> for ObserVec<N> {
    fn from(value: SVector<f32, N>) -> Self {
        ObserVec::from(value.data.0[0])
    }
}

impl<const N: usize> Mul<f32> for ObserVec<N> {
    type Output = ObserVec<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        ObserVec::<N>(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<f32> for &ObserVec<N> {
    type Output = ObserVec<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        ObserVec::<N>(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ObserVec<N>> for f32 {
    type Output = ObserVec<N>;

    fn mul(self, rhs: ObserVec<N>) -> Self::Output {
        ObserVec::<N>(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ObserVec<N>> for f32 {
    type Output = ObserVec<N>;

    fn mul(self, rhs: &'a ObserVec<N>) -> Self::Output {
        ObserVec::<N>(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> OcnusObser for ObserVec<N> {}

impl<const N: usize> PartialEq for ObserVec<N> {
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .map(|(val_x, val_y)| val_x == val_y)
            .fold(true, |acc, next| acc & next)
    }
}

impl<const N: usize> Sub for ObserVec<N> {
    type Output = ObserVec<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        ObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 - v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> SubAssign for ObserVec<N> {
    fn sub_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= rhs);
    }
}

impl<'a, const N: usize> Sub<&'a ObserVec<N>> for &'a ObserVec<N> {
    type Output = ObserVec<N>;

    fn sub(self, rhs: &'a ObserVec<N>) -> Self::Output {
        ObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| *v1 - *v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Zero for ObserVec<N> {
    fn is_zero(&self) -> bool {
        self.0.iter().all(|value| value == &0.0)
    }

    fn set_zero(&mut self) {
        self.0 = [0.0; N]
    }

    fn zero() -> Self {
        Self([0.0; N])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obsvec() {
        let ov_d = ObserVec::<3>::default();

        assert!(ov_d.0[0].is_nan());
        assert!(ov_d.0[1].is_nan());
        assert!(ov_d.0[2].is_nan());

        let ov_1 = ObserVec::<3>::from([1.0, 2.0, 3.0]);

        assert!(ov_1.ss() == 14.0);

        let ov_2 = ObserVec::<3>::from([2.0, 3.0, 1.0]);

        assert!((ov_1.clone() + ov_2.clone()).ss() == 9.0 + 25.0 + 16.0);
        assert!((ov_1.clone() - ov_2.clone()).ss() == 6.0);
        assert!(ov_1.mse(&ov_2) == 2.0);

        assert!((&ov_1 + &ov_2).ss() == 9.0 + 25.0 + 16.0);
        assert!((&ov_1 - &ov_2).ss() == 6.0);

        assert!((ov_1.clone() * 2.0).ss() == 4.0 * 14.0);
        assert!((2.0 * ov_1.clone()).ss() == 4.0 * 14.0);

        assert!((&ov_1 * 2.0).ss() == 4.0 * 14.0);
        assert!((2.0 * &ov_1).ss() == 4.0 * 14.0);

        assert!(ObserVec::<3>::default().any_nan());
        assert!(ObserVec::<3>::default().is_nan());

        assert!(ObserVec([f32::NAN, 0.0, 0.0]).any_nan());
        assert!(!ObserVec([f32::NAN, 0.0, 0.0]).is_nan());

        assert!(ObserVec([0.0, f32::NAN, 0.0, 0.0]).any_nan());
        assert!(!ObserVec([0.0, f32::NAN, 0.0, 0.0]).is_nan());

        assert!(ObserVec([0.0, f32::NAN, 0.0, 0.0]) != ObserVec([0.0, f32::NAN, 0.0, 0.0]));
        assert!(ObserVec::<4>::zeros() == ObserVec::<4>::zeros());

        assert!(ObserVec::<4>::default().is_nan());
    }
}
