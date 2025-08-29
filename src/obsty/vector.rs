use crate::{
    base::ScObs,
    math::CovMatrix,
    obsty::{NoiseModel, Observable},
};
use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::{Const, DVector, Dyn, OVector, RealField, SVector};
use num_traits::Zero;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, ops::Div};
use std::{
    fmt::{Debug, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

/// A newtype for storing coordinates and basis vectors.
#[derive(
    derive_more::Add,
    Clone,
    Debug,
    Deref,
    Deserialize,
    From,
    Index,
    IndexMut,
    IntoIterator,
    PartialEq,
    Serialize,
)]
#[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
pub struct ICSCoordsBasis<T>(ObserVec<T, 12>)
where
    T: RealField;

impl<T> From<[T; 12]> for ICSCoordsBasis<T>
where
    T: RealField,
{
    fn from(value: [T; 12]) -> Self {
        Self(ObserVec::from(value))
    }
}

impl<T> Zero for ICSCoordsBasis<T>
where
    T: Copy + RealField,
{
    fn is_zero(&self) -> bool {
        self.0.iter().all(|value| value == &T::zero())
    }

    fn set_zero(&mut self) {
        self.0 = ObserVec::zeros();
    }

    fn zero() -> Self {
        Self(ObserVec::<T, 12>::zeros())
    }
}

/// Generic N-dimensional observation vector.
#[derive(
    Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, PartialEq, Serialize,
)]
#[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
pub struct ObserVec<T, const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [T; N],
);

impl<T, const N: usize> ObserVec<T, N>
where
    T: Copy + RealField,
{
    /// Calculate the mean square error between two observation vectors.
    ///
    /// If both observations are not considered valid, returns `0.0`.
    /// If only one of the two observations is considered valid, returns `NaN`.
    pub fn mean_square_error(&self, other: &Self) -> T
    where
        T: Sum,
        Self: Observable,
    {
        if self.is_valid() & other.is_valid() {
            (self - other).sum_of_squares() / T::from_usize(N).unwrap()
        } else if !self.is_valid() & !other.is_valid() {
            T::zero()
        } else {
            (-T::one()).sqrt()
        }
    }

    /// Calculate the norm of the vector.
    pub fn norm(&self) -> T {
        SVector::from(self.0).norm()
    }

    /// Return a normalized vector.
    pub fn normalize(&self) -> Self {
        Self::from(
            self.0
                .iter()
                .map(|value| *value / SVector::from(self.0).norm())
                .collect::<Vec<T>>()
                .as_slice(),
        )
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
    T: RealField,
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
    T: Copy + RealField,
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
    T: Copy + RealField,
{
    fn add_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += *rhs);
    }
}

impl<'a, T, const N: usize> AddAssign<&'a ObserVec<T, N>> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn add_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += *rhs);
    }
}

impl<T, const N: usize> Default for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn default() -> Self {
        ObserVec([(-T::one()).sqrt(); N])
    }
}

impl<T, const N: usize> Div<T> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    type Output = ObserVec<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|value| *value / rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Div<T> for &ObserVec<T, N>
where
    T: Copy + RealField,
{
    type Output = ObserVec<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|value| *value / rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T> Display for ObserVec<T, 3>
where
    T: Copy + RealField,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl<T> Display for ObserVec<T, 4>
where
    T: RealField,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} | {:.2}]",
            self[0], self[1], self[2], self[3]
        )
    }
}

impl<T, const N: usize> From<&[T]> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn from(value: &[T]) -> Self {
        let mut array = [T::zero(); N];

        assert!(
            value.len() == N,
            "failed ObserVec conversion, invalid slice length"
        );

        array
            .iter_mut()
            .zip(value.iter().take(N))
            .for_each(|(target, value)| *target = *value);

        ObserVec::from(array)
    }
}

impl<T, const N: usize> From<SVector<T, N>> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn from(value: SVector<T, N>) -> Self {
        ObserVec::from(value.data.0[0])
    }
}

impl<T, const N: usize> Mul<T> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    type Output = ObserVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|value| *value * rhs)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Mul<T> for &ObserVec<T, N>
where
    T: Copy + RealField,
{
    type Output = ObserVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObserVec(
            self.iter()
                .map(|value| *value * rhs)
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
                .map(|value| *value * self)
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
                .map(|value| *value * self)
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
                .map(|value| *value * self)
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
                .map(|value| *value * self)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Observable for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn is_valid(&self) -> bool {
        self.iter().fold(true, |acc, next| acc & next.is_finite())
    }
}

impl<T, const N: usize> Sub for ObserVec<T, N>
where
    T: RealField,
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
    T: Copy + RealField,
{
    fn sub_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= *rhs);
    }
}

impl<'a, T, const N: usize> SubAssign<&'a ObserVec<T, N>> for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= *rhs);
    }
}

impl<'a, T, const N: usize> Sub<&'a ObserVec<T, N>> for &'a ObserVec<T, N>
where
    T: Copy + RealField,
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
    T: Copy + RealField,
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
pub enum ObserVecNoise<T>
where
    T: RealField,
{
    Gaussian(T, u64),
    Multivariate(CovMatrix<T, Dyn>, u64),
}

impl<T, const N: usize> NoiseModel<T, ObserVec<T, N>> for ObserVecNoise<T>
where
    T: Copy + RealField,
    StandardNormal: Distribution<T>,
{
    fn generate_noise(
        &self,
        scobs: &ScObs<T, ObserVec<T, N>>,
        rng: &mut impl rand::Rng,
    ) -> nalgebra::DVector<ObserVec<T, N>> {
        match self {
            ObserVecNoise::Gaussian(std_dev, ..) => {
                let normal = StandardNormal;
                let size = scobs.len();

                DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal) * *std_dev; N])),
                )
            }
            ObserVecNoise::Multivariate(covmat, ..) => {
                let normal = StandardNormal;
                let size = scobs.len();

                let mut result = DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal); N])),
                );

                for i in 0..N {
                    let values = covmat.l().unwrap()
                        * OVector::<T, Const<N>>::from_iterator(
                            (0..size).map(|_| rng.sample(normal)),
                        );

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

/// Normalized chi-squared error metric for the [`ObserVec`] type, as defined in Nieves-Chinchilla et al. (2019).
pub fn observec_nchisq<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    let total_error = obser
        .iter()
        .zip(other)
        .map(|(out_vec, ref_vec)| {
            if !ref_vec.is_valid() && !out_vec.is_valid() {
                T::zero()
            } else if ref_vec.is_valid() && out_vec.is_valid() {
                (ref_vec - out_vec).sum_of_squares()
                    + (ref_vec.iter().map(|value| value.powi(2)).sum::<T>().sqrt()
                        - out_vec.iter().map(|value| value.powi(2)).sum::<T>().sqrt())
                    .powi(2)
            } else {
                T::one() / T::zero()
            }
        })
        .sum::<T>();

    // Correct for double NaN's
    let normalizer = obser
        .iter()
        .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

    let b_max = obser.iter().fold(T::zero(), |acc, value| {
        match acc.partial_cmp(&value.norm()) {
            Some(Ordering::Less) => value.norm(),
            _ => acc,
        }
    });

    total_error / T::from_usize(normalizer).unwrap() / b_max.powi(2)
}

/// Mean square error (MSE) for the [`ObserVec`] type.
pub fn observec_mse<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    // Correct for double NaN's
    let normalizer = obser
        .iter()
        .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

    obser
        .iter()
        .zip(other)
        .map(|(out_vec, ref_vec)| {
            if !ref_vec.is_valid() && !out_vec.is_valid() {
                T::zero()
            } else if ref_vec.is_valid() && out_vec.is_valid() {
                (ref_vec - out_vec).sum_of_squares()
            } else {
                T::one() / T::zero()
            }
        })
        .sum::<T>()
        / T::from_usize(normalizer * N).unwrap()
}

/// Mean square error (MSE) percentage for the [`ObserVec`] type.
pub fn observec_msep<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    // Correct for double NaN's
    let normalize_vector = DVector::from_iterator(
        obser.len(),
        obser.iter().map(|obs| {
            if obs.is_valid() {
                ObserVec::<T, N>::zeros()
            } else {
                ObserVec::default()
            }
        }),
    );

    observec_mse(obser, other) / observec_mse(normalize_vector.as_slice(), other)
}

/// Root mean square error (RMSE) for the [`ObserVec`] type.
pub fn observec_rmse<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    observec_mse(obser, other).sqrt()
}

/// Root mean square error (RMSE) percentage for the [`ObserVec`] type.
pub fn observec_rmsep<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    let normalize_vector = DVector::from_iterator(
        obser.len(),
        obser.iter().map(|obs| {
            if obs.is_valid() {
                ObserVec::<T, N>::zeros()
            } else {
                ObserVec::default()
            }
        }),
    );

    (observec_mse(obser, other) / observec_mse(normalize_vector.as_slice(), other)).sqrt()
}

/// Mean square error (MSE), with absolute value included, for the [`ObserVec`] type.
pub fn observec_mset<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    // Correct for double NaN's
    let normalizer = obser
        .iter()
        .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

    obser
        .iter()
        .zip(other)
        .map(|(out_vec, ref_vec)| {
            if !ref_vec.is_valid() && !out_vec.is_valid() {
                T::zero()
            } else if ref_vec.is_valid() && out_vec.is_valid() {
                (ref_vec - out_vec).sum_of_squares()
                    + (ref_vec.iter().map(|value| value.powi(2)).sum::<T>().sqrt()
                        - out_vec.iter().map(|value| value.powi(2)).sum::<T>().sqrt())
                    .powi(2)
            } else {
                T::one() / T::zero()
            }
        })
        .sum::<T>()
        / T::from_usize(normalizer * N).unwrap()
}

/// Root mean square error (RMSE), with absolute value included, for the [`ObserVec`] type.
pub fn observec_rmset<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    observec_mset(obser, other).sqrt()
}

/// Root mean square error (RMSE), with absolute value included, percentage for the [`ObserVec`] type.
pub fn observec_rmsetp<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> T
where
    T: Copy + RealField + Sum,
{
    let normalize_vector = DVector::from_iterator(
        obser.len(),
        obser.iter().map(|obs| {
            if obs.is_valid() {
                ObserVec::<T, N>::zeros()
            } else {
                ObserVec::default()
            }
        }),
    );

    (observec_mset(obser, other) / observec_mset(normalize_vector.as_slice(), other)).sqrt()
}

/// Checks for validity in between two [`ObserVec`] types.
pub fn observec_valid<T, const N: usize>(obser: &[ObserVec<T, N>], other: &[ObserVec<T, N>]) -> bool
where
    T: Copy + RealField + Sum,
{
    obser
        .iter()
        .zip(other)
        .fold(true, |acc, (out_vec, ref_vec)| {
            if (!ref_vec.is_valid() && !out_vec.is_valid())
                | (ref_vec.is_valid() && out_vec.is_valid())
            {
                acc & true
            } else {
                acc & false
            }
        })
}

#[cfg(test)]
mod tests {
    use approx::ulps_eq;
    use nalgebra::DMatrix;

    use super::*;

    #[test]
    fn test_observec() {
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

        assert!((ov_1.clone() / 2.0).sum_of_squares() == 14.0 / 4.0);
        assert!((&ov_1 / 2.0).sum_of_squares() == 14.0 / 4.0);

        assert!(!ObserVec::<f64, 3>::default().is_valid());

        assert!(!ObserVec([f64::NAN, 0.0, 0.0]).is_valid());

        assert!(!ObserVec([0.0, f64::NAN, 0.0, 0.0]).is_valid());
    }

    #[test]
    fn test_observec_metrics() {
        let array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([1.0, 2.0, 3.0])));

        let array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        assert!(ulps_eq!(
            observec_mse(array_1.as_slice(), array_2.as_slice()),
            2.0
        ));
        assert!(ulps_eq!(
            observec_rmse(array_1.as_slice(), array_2.as_slice()),
            2.0_f64.sqrt()
        ));

        assert!(ulps_eq!(
            observec_msep(array_1.as_slice(), array_2.as_slice()),
            2.0 / 14.0 * 3.0
        ));

        assert!(observec_valid(array_1.as_slice(), array_2.as_slice()));
    }

    #[test]
    fn test_observec_invalids() {
        let mut array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([1.0, 2.0, 3.0])));

        array_1[0] = ObserVec::default();

        let mut array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        assert!(!observec_valid(array_1.as_slice(), array_2.as_slice()));

        array_2[0] = ObserVec::default();

        assert!(observec_valid(array_1.as_slice(), array_2.as_slice()));

        array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([1.0, 2.0, 3.0])));
        array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        array_1[4] = ObserVec::default();
        array_2[3] = ObserVec::default();
        array_2[4] = ObserVec::default();

        assert!(!observec_valid(array_1.as_slice(), array_2.as_slice()));

        array_1[0] = ObserVec::default();
        array_2[0] = ObserVec::default();
        array_1[3] = ObserVec::default();

        assert!(observec_valid(array_1.as_slice(), array_2.as_slice()));

        array_1[2] = ObserVec::default();

        assert!(!observec_valid(array_1.as_slice(), array_2.as_slice()));
    }

    #[test]
    fn test_observec_likelihood() {
        let array_1 = DVector::from_iterator(
            7,
            (0..7).map(|idx| {
                if (idx == 0) || (idx == 6) {
                    ObserVec::default()
                } else {
                    ObserVec::<f32, 3>::from([0.1, -0.05, 0.11])
                }
            }),
        );

        let array_2 = DVector::from_iterator(
            7,
            (0..7).map(|idx| {
                if (idx == 0) || (idx == 6) {
                    ObserVec::default()
                } else {
                    ObserVec::<f32, 3>::zeros()
                }
            }),
        );

        let covm = CovMatrix::new(DMatrix::from_diagonal_element(7, 7, 1.0), true).unwrap();

        let ll = covm.observec_log_likelihood(array_1.as_slice(), array_2.as_slice());

        assert!(ulps_eq!(ll, -13.845578));
    }
}
