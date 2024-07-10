//! Implementation of a `N`-dimensional vector observable.

use crate::{
    fXX,
    math::{CovMatrix, powi},
    obser::OcnusObser,
};
use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::{DMatrix, DVector, DVectorView, SVector};
use num_traits::{Float, Zero};
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

use super::{OcnusNoise, ScObsSeries};

/// Generic N-dimensional observation vector.
#[derive(Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, Serialize)]
pub struct ObserVec<T, const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    #[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
    [T; N],
);

impl<T, const N: usize> ObserVec<T, N>
where
    T: fXX,
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
    pub fn sum_of_squares(&self) -> T {
        self.iter().map(|value| powi!(*value, 2)).sum::<T>()
    }

    /// Return a new observation vector filled with zeros.
    pub fn zeros() -> Self {
        ObserVec([T::zero(); N])
    }
}

impl<T, const N: usize> Add for ObserVec<T, N>
where
    T: fXX,
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
    T: fXX,
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
    T: fXX,
{
    fn add_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += *rhs);
    }
}

impl<'a, T, const N: usize> AddAssign<&'a ObserVec<T, N>> for ObserVec<T, N>
where
    T: fXX,
{
    fn add_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += *rhs);
    }
}

impl<T, const N: usize> Default for ObserVec<T, N>
where
    T: fXX,
{
    fn default() -> Self {
        ObserVec([T::nan(); N])
    }
}

impl<T> Display for ObserVec<T, 3>
where
    T: fXX,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl<T> Display for ObserVec<T, 4>
where
    T: fXX,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} | {:.2}]",
            self[0], self[1], self[2], self[3]
        )
    }
}

impl<T> Display for ObserVec<T, 12>
where
    T: fXX,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} - {:.2} | {:.2} | {:.2} - {:.2} | {:.2} | {:.2} - {:.2} | {:.2} | {:.2}]",
            self[0],
            self[1],
            self[2],
            self[3],
            self[4],
            self[5],
            self[6],
            self[7],
            self[8],
            self[9],
            self[10],
            self[11]
        )
    }
}

impl<T, const N: usize> From<SVector<T, N>> for ObserVec<T, N>
where
    T: fXX,
{
    fn from(value: SVector<T, N>) -> Self {
        ObserVec::from(value.data.0[0])
    }
}

impl<T, const N: usize> Mul<T> for ObserVec<T, N>
where
    T: fXX,
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
    T: fXX,
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
    T: fXX,
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
    T: fXX,
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
    T: fXX,
{
    fn sub_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= *rhs);
    }
}

impl<'a, T, const N: usize> SubAssign<&'a ObserVec<T, N>> for ObserVec<T, N>
where
    T: fXX,
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= *rhs);
    }
}

impl<'a, T, const N: usize> Sub<&'a ObserVec<T, N>> for &'a ObserVec<T, N>
where
    T: fXX,
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
    T: fXX,
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

/// Generic N-dimensional observation vector noise
#[derive(Clone, Debug, Deserialize, Serialize)]
#[allow(missing_docs)]
pub enum ObserVecNoise<T, const N: usize>
where
    T: fXX,
{
    Gaussian(T, u64),
    Multivariate(CovMatrix<T>, u64),
}

impl<T, const N: usize> ObserVecNoise<T, N>
where
    T: fXX,
{
    /// Compute the likelihood for an observation `x` with expected mean `mu`.
    pub fn multivariate_likelihood(
        &self,
        x: &DVectorView<ObserVec<T, N>>,
        mu: &ScObsSeries<T, ObserVec<T, N>>,
    ) -> T {
        let mut x_flat = x
            .iter()
            .flat_map(|x_obs| x_obs.iter().cloned().collect::<Vec<T>>())
            .collect::<Vec<T>>();

        let mut mu_flat = mu
            .into_iter()
            .flat_map(|mu_scobs| {
                mu_scobs
                    .get_observation()
                    .iter()
                    .cloned()
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<T>>();

        // Correct matching NaN's
        x_flat
            .iter_mut()
            .zip(mu_flat.iter_mut())
            .for_each(|(xv, muv)| {
                if xv.is_nan() && muv.is_nan() {
                    *xv = T::zero();
                    *muv = T::zero();
                }
            });

        match self {
            ObserVecNoise::Gaussian(std_dev, ..) => {
                let covmat = CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(x.len(), x.len(), *std_dev).as_view(),
                )
                .unwrap();

                covmat.multivariate_likelihood(x_flat, mu_flat)
            }
            ObserVecNoise::Multivariate(covmat, ..) => {
                covmat.multivariate_likelihood(x_flat, mu_flat)
            }
        }
    }
}

impl<T, const N: usize> OcnusNoise<T, ObserVec<T, N>> for ObserVecNoise<T, N>
where
    T: fXX,
    StandardNormal: Distribution<T>,
{
    fn generate_noise(
        &self,
        series: &super::ScObsSeries<T, ObserVec<T, N>>,
        rng: &mut impl rand::Rng,
    ) -> nalgebra::DVector<ObserVec<T, N>> {
        match self {
            ObserVecNoise::Gaussian(std_dev, ..) => {
                let normal = Normal::new(T::zero(), *std_dev).unwrap();
                let size = series.len();

                DVector::from_iterator(size, (0..size).map(|_| ObserVec([rng.sample(normal); N])))
            }
            ObserVecNoise::Multivariate(covmat, ..) => {
                let normal = Normal::new(T::zero(), T::one()).unwrap();
                let size = series.len();

                let mut result = DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal); N])),
                );

                for i in 0..N {
                    let values = covmat.ref_cholesky_ltm()
                        * DVector::from_iterator(size, (0..size).map(|_| rng.sample(normal)));

                    result
                        .iter_mut()
                        .zip(values.row_iter())
                        .for_each(|(res, val)| res.0[i] = val[(0, 0)]);
                }

                result
            }
        }
    }

    fn get_random_seed(&self) -> u64 {
        match self {
            ObserVecNoise::Gaussian(.., seed) => *seed,
            ObserVecNoise::Multivariate(.., seed) => *seed,
        }
    }

    fn increment_random_seed(&mut self) {
        match self {
            ObserVecNoise::Gaussian(.., seed) => {
                *seed += 1;
            }
            ObserVecNoise::Multivariate(.., seed) => {
                *seed += 1;
            }
        }
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
