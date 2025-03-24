//! Implementation of a `N`-dimensional vector observable.

use crate::obser::OcnusObser;
use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::SVector;
use num_traits::{Float, FromPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

/// Generic N-dimensional observation vector.
#[derive(Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, Serialize)]
pub struct ObserVec<T, const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    #[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
    pub [T; N],
);

impl<T, const N: usize> ObserVec<T, N>
where
    T: Float + FromPrimitive,
{
    /// Returns true if any entry within the observation vector is `NaN`.
    pub fn any_nan(&self) -> bool {
        !self.iter().fold(true, |acc, next| acc & next.is_finite())
    }

    /// Returns true if all entries within the observation vector are `NaN`.
    pub fn is_nan(&self) -> bool {
        self.iter().fold(true, |acc, next| acc & next.is_nan())
    }

    /// Calculate the mean square error between two observation vectors.
    ///
    /// If both observations are not considered valid, returns `0.0`.
    /// If only one of the two observations is considered valid, returns `NaN`.
    pub fn mean_square_error(&self, other: &Self) -> T
    where
        T: 'static + Copy + Debug + Sum,
        Self: OcnusObser,
    {
        if self.is_valid() & other.is_valid() {
            (self - other).sum_of_squares() / T::from_usize(N).unwrap()
        } else if !self.is_valid() & !other.is_valid() {
            T::zero()
        } else {
            T::nan()
        }
    }

    /// Calculate the sum of squares over the entries within the observation vector.
    pub fn sum_of_squares(&self) -> T
    where
        T: Sum,
    {
        self.iter().map(|value| value.powi(2)).sum::<T>()
    }

    /// Return a new observation vector filled with zeros.
    pub fn zeros() -> Self {
        ObserVec([T::zero(); N])
    }
}

impl<T, const N: usize> Add for ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn add(self, rhs: Self) -> Self::Output {
        ObserVec(
            zip_eq(self, rhs)
                .map(|(a, b)| a + b)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, T, const N: usize> Add<&'a ObserVec<T, N>> for &'a ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn add(self, rhs: &'a ObserVec<T, N>) -> Self::Output {
        ObserVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| *v1 + *v2)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> AddAssign for ObserVec<T, N>
where
    T: for<'x> AddAssign<&'x T> + Float,
{
    fn add_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += rhs);
    }
}

impl<T, const N: usize> Default for ObserVec<T, N>
where
    T: Float,
{
    fn default() -> Self {
        ObserVec([T::nan(); N])
    }
}

impl<T> Display for ObserVec<T, 3>
where
    T: Display + Float,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl<T> Display for ObserVec<T, 4>
where
    T: Display + Float,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} | {:.2}]",
            self[0], self[1], self[2], self[3]
        )
    }
}

impl<T, const N: usize> From<SVector<T, N>> for ObserVec<T, N>
where
    T: Float,
{
    fn from(value: SVector<T, N>) -> Self {
        ObserVec::from(value.data.0[0])
    }
}

impl<T, const N: usize> Mul<T> for ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Mul<T> for &ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ObserVec<f32, N>> for f32 {
    type Output = ObserVec<f32, N>;

    fn mul(self, rhs: ObserVec<f32, N>) -> Self::Output {
        ObserVec(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ObserVec<f64, N>> for f64 {
    type Output = ObserVec<f64, N>;

    fn mul(self, rhs: ObserVec<f64, N>) -> Self::Output {
        ObserVec(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ObserVec<f32, N>> for f32 {
    type Output = ObserVec<f32, N>;

    fn mul(self, rhs: &'a ObserVec<f32, N>) -> Self::Output {
        ObserVec(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ObserVec<f64, N>> for f64 {
    type Output = ObserVec<f64, N>;

    fn mul(self, rhs: &'a ObserVec<f64, N>) -> Self::Output {
        ObserVec(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> OcnusObser for ObserVec<T, N>
where
    T: Copy + Default + Float + FromPrimitive + Send + Sync,
{
    fn is_valid(&self) -> bool {
        !self.any_nan()
    }
}

impl<T, const N: usize> PartialEq for ObserVec<T, N>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .map(|(val_x, val_y)| val_x == val_y)
            .fold(true, |acc, next| acc & next)
    }
}

impl<T, const N: usize> Sub for ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        ObserVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 - v2)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> SubAssign for ObserVec<T, N>
where
    T: Float + for<'x> SubAssign<&'x T>,
{
    fn sub_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= rhs);
    }
}

impl<'a, T, const N: usize> Sub<&'a ObserVec<T, N>> for &'a ObserVec<T, N>
where
    T: Debug + Float,
{
    type Output = ObserVec<T, N>;

    fn sub(self, rhs: &'a ObserVec<T, N>) -> Self::Output {
        ObserVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| *v1 - *v2)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Zero for ObserVec<T, N>
where
    T: Debug + Float,
{
    fn is_zero(&self) -> bool {
        self.0.iter().all(|value| value == &T::zero())
    }

    fn set_zero(&mut self) {
        self.0 = [T::zero(); N]
    }

    fn zero() -> Self {
        Self([T::zero(); N])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observec() {
        let ov_d = ObserVec::<f64, 3>::default();

        assert!(ov_d.0[0].is_nan());
        assert!(ov_d.0[1].is_nan());
        assert!(ov_d.0[2].is_nan());

        let ov_1 = ObserVec::<f64, 3>::from([1.0, 2.0, 3.0]);

        assert!(ov_1.sum_of_squares() == 14.0);

        let ov_2 = ObserVec::<f64, 3>::from([2.0, 3.0, 1.0]);

        assert!((ov_1.clone() + ov_2.clone()).sum_of_squares() == 9.0 + 25.0 + 16.0);
        assert!((ov_1.clone() - ov_2.clone()).sum_of_squares() == 6.0);
        assert!(ov_1.mean_square_error(&ov_2) == 2.0);

        assert!((&ov_1 + &ov_2).sum_of_squares() == 9.0 + 25.0 + 16.0);
        assert!((&ov_1 - &ov_2).sum_of_squares() == 6.0);

        assert!((ov_1.clone() * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * ov_1.clone()).sum_of_squares() == 4.0 * 14.0);

        assert!((&ov_1 * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * &ov_1).sum_of_squares() == 4.0 * 14.0);

        assert!(ObserVec::<f64, 3>::default().any_nan());
        assert!(ObserVec::<f64, 3>::default().is_nan());

        assert!(ObserVec([f64::NAN, 0.0, 0.0]).any_nan());
        assert!(!ObserVec([f64::NAN, 0.0, 0.0]).is_nan());

        assert!(ObserVec([0.0, f64::NAN, 0.0, 0.0]).any_nan());
        assert!(!ObserVec([0.0, f64::NAN, 0.0, 0.0]).is_nan());

        assert!(ObserVec([0.0, f64::NAN, 0.0, 0.0]) != ObserVec([0.0, f64::NAN, 0.0, 0.0]));
        assert!(ObserVec::<f64, 4>::zeros() == ObserVec::<f64, 4>::zeros());

        assert!(ObserVec::<f64, 4>::default().is_nan());
    }
}
