use crate::obs::data::ObsData;
use derive_more::{Deref, DerefMut, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::{DVector, RealField, SVector};
use num_traits::{AsPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, ops::Div};
use std::{
    fmt::{Debug, Display, Formatter},
    iter::Sum,
    ops::{Add, AddAssign, Mul, Sub, SubAssign},
};

/// Generic N-dimensional observation vector.
#[derive(
    Clone,
    Debug,
    Deref,
    DerefMut,
    Deserialize,
    From,
    Index,
    IndexMut,
    IntoIterator,
    PartialEq,
    Serialize,
)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct ObsVec<T, const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [T; N],
);

impl<T, const N: usize> ObsVec<T, N>
where
    T: RealField,
{
    /// Calculate the mean square error between two observation vectors.
    ///
    /// If both observations are not considered valid, returns `0.0`.
    /// If only one of the two observations is considered valid, returns `NaN`.
    pub fn mean_square_error(&self, other: &Self) -> T
    where
        T: Sum,
        Self: ObsData,
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
        SVector::from(self.0.clone()).norm()
    }

    /// Return a normalized vector.
    pub fn normalize(&self) -> Self {
        Self::from(
            self.0
                .iter()
                .map(|value| value.clone() / SVector::from(self.0.clone()).norm())
                .collect::<Vec<T>>()
                .as_slice(),
        )
    }

    /// S
    pub fn set(&mut self, index: usize, value: T) {
        self.0[index] = value;
    }

    /// Calculate the sum of squares over the entries within the observation vector.
    pub fn sum_of_squares(&self) -> T
    where
        T: Sum,
    {
        self.iter().map(|value| value.clone().powi(2)).sum::<T>()
    }

    /// Return a new observation vector filled with zeros.
    pub fn zeros() -> Self {
        ObsVec(core::array::from_fn(|_| T::zero()))
    }
}

impl<T, const N: usize> Add for ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn add(self, rhs: Self) -> Self::Output {
        ObsVec(
            zip_eq(self, rhs)
                .map(|(a, b)| a + b)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, T, const N: usize> Add<&'a ObsVec<T, N>> for &'a ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn add(self, rhs: &'a ObsVec<T, N>) -> Self::Output {
        ObsVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1.clone() + v2.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> AddAssign for ObsVec<T, N>
where
    T: RealField,
{
    fn add_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += rhs.clone());
    }
}

impl<'a, T, const N: usize> AddAssign<&'a ObsVec<T, N>> for ObsVec<T, N>
where
    T: RealField,
{
    fn add_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value += rhs.clone());
    }
}

impl<T, const N: usize> Default for ObsVec<T, N>
where
    T: RealField,
{
    fn default() -> Self {
        ObsVec(core::array::from_fn(|_| (-T::one()).sqrt()))
    }
}

impl<T, const N: usize> Div<T> for ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        ObsVec(
            self.iter()
                .map(|value| value.clone() / rhs.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Div<T> for &ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        ObsVec(
            self.iter()
                .map(|value| value.clone() / rhs.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T> Display for ObsVec<T, 3>
where
    T: RealField,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl<T> Display for ObsVec<T, 4>
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

impl<T, const N: usize> From<&[T]> for ObsVec<T, N>
where
    T: RealField,
{
    fn from(value: &[T]) -> Self {
        let mut array = core::array::from_fn(|_| T::zero());

        assert!(
            value.len() == N,
            "failed ObsVec conversion, invalid slice length"
        );

        array
            .iter_mut()
            .zip(value.iter().take(N))
            .for_each(|(target, value)| *target = value.clone());

        ObsVec::from(array)
    }
}

impl<T, const N: usize> From<SVector<T, N>> for ObsVec<T, N>
where
    T: RealField,
{
    fn from(value: SVector<T, N>) -> Self {
        ObsVec::from(value.data.0[0].clone())
    }
}

impl<T, const N: usize> Mul<T> for ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObsVec(
            self.iter()
                .map(|value| value.clone() * rhs.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Mul<T> for &ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        ObsVec(
            self.iter()
                .map(|value| value.clone() * rhs.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ObsVec<f32, N>> for f32 {
    type Output = ObsVec<f32, N>;

    fn mul(self, rhs: ObsVec<f32, N>) -> Self::Output {
        ObsVec(
            rhs.iter()
                .map(|value| *value * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ObsVec<f64, N>> for f64 {
    type Output = ObsVec<f64, N>;

    fn mul(self, rhs: ObsVec<f64, N>) -> Self::Output {
        ObsVec(
            rhs.iter()
                .map(|value| *value * self)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ObsVec<f32, N>> for f32 {
    type Output = ObsVec<f32, N>;

    fn mul(self, rhs: &'a ObsVec<f32, N>) -> Self::Output {
        ObsVec(
            rhs.iter()
                .map(|value| *value * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ObsVec<f64, N>> for f64 {
    type Output = ObsVec<f64, N>;

    fn mul(self, rhs: &'a ObsVec<f64, N>) -> Self::Output {
        ObsVec(
            rhs.iter()
                .map(|value| *value * self)
                .collect::<Vec<f64>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> ObsData for ObsVec<T, N>
where
    T: RealField,
{
    fn is_valid(&self) -> bool {
        self.iter().fold(true, |acc, next| acc & next.is_finite())
    }
}

impl<T, const N: usize> Sub for ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn sub(self, rhs: Self) -> Self::Output {
        ObsVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 - v2)
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> SubAssign for ObsVec<T, N>
where
    T: RealField,
{
    fn sub_assign(&mut self, rhs: Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= rhs.clone());
    }
}

impl<'a, T, const N: usize> SubAssign<&'a ObsVec<T, N>> for ObsVec<T, N>
where
    T: RealField,
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        zip_eq(self.0.iter_mut(), rhs.iter()).for_each(|(value, rhs)| *value -= rhs.clone());
    }
}

impl<'a, T, const N: usize> Sub<&'a ObsVec<T, N>> for &'a ObsVec<T, N>
where
    T: RealField,
{
    type Output = ObsVec<T, N>;

    fn sub(self, rhs: &'a ObsVec<T, N>) -> Self::Output {
        ObsVec(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1.clone() - v2.clone())
                .collect::<Vec<T>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<T, const N: usize> Zero for ObsVec<T, N>
where
    T: RealField,
{
    fn is_zero(&self) -> bool {
        self.0.iter().all(|value| value == &T::zero())
    }

    fn set_zero(&mut self) {
        self.0 = core::array::from_fn(|_| T::zero());
    }

    fn zero() -> Self {
        Self(core::array::from_fn(|_| T::zero()))
    }
}

/// Metric types for comparing two [`ObsVec`] slices.
pub enum ObsVecMetric {
    /// Mean squared error (MSE).
    MSE,
    /// Mean squared percentage error (MSE).
    MSPE,
    /// Root mean square error (RMSE).
    RMSE,
    /// Root mean square percentage error (RMSPE).
    RMSPE,
    /// Normalized mean squared error (NMSE).
    NMSE,
    /// Normalized root mean square error (NRMSE).
    NRMSE,
    /// Normalized chi-squared error metric, as used in Nieves-Chinchilla et al. (2019).
    NChiSq,
    /// Dynamic Time Warping (DTW) metric.
    DTW,
    /// Validity check between two observation vector slices.
    /// Returns 0 if valid, infinity if invalid.
    Valid,
}

/// Returns the error for a specific metric from two [`ObsVec`] slices.
pub fn ov_error<T, const N: usize>(
    x: &[ObsVec<T, N>],
    y: &[ObsVec<T, N>],
    metric: ObsVecMetric,
) -> T
where
    T: AsPrimitive<f32> + RealField + Sum,
{
    match metric {
        ObsVecMetric::MSE => {
            // Correct for double NaN's
            let normalizer = x
                .iter()
                .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

            x.iter()
                .zip(y)
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
        ObsVecMetric::MSPE => {
            // Correct for double NaN's
            let normalizer = x
                .iter()
                .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

            x.iter()
                .zip(y)
                .map(|(out_vec, ref_vec)| {
                    if !ref_vec.is_valid() && !out_vec.is_valid() {
                        T::zero()
                    } else if ref_vec.is_valid() && out_vec.is_valid() {
                        (ref_vec - out_vec).sum_of_squares() / ref_vec.sum_of_squares()
                    } else {
                        T::one() / T::zero()
                    }
                })
                .sum::<T>()
                / T::from_usize(normalizer * N).unwrap()
        }
        ObsVecMetric::RMSE => ov_error(x, y, ObsVecMetric::MSE).sqrt(),
        ObsVecMetric::RMSPE => ov_error(x, y, ObsVecMetric::MSPE).sqrt(),
        ObsVecMetric::NMSE => {
            let normalize_vector = DVector::from_iterator(
                x.len(),
                x.iter().map(|obs| {
                    if obs.is_valid() {
                        ObsVec::<T, N>::zeros()
                    } else {
                        ObsVec::default()
                    }
                }),
            );

            ov_error(x, y, ObsVecMetric::MSE)
                / ov_error(normalize_vector.as_slice(), y, ObsVecMetric::MSE)
        }
        ObsVecMetric::NRMSE => ov_error(x, y, ObsVecMetric::NMSE).sqrt(),
        ObsVecMetric::NChiSq => {
            let total_error = x
                .iter()
                .zip(y)
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
            let normalizer = x
                .iter()
                .fold(0, |acc, next| if next.is_valid() { acc + 1 } else { acc });

            let b_max = x.iter().fold(T::zero(), |acc, value| {
                match acc.partial_cmp(&value.norm()) {
                    Some(Ordering::Less) => value.norm(),
                    _ => acc,
                }
            });

            total_error / T::from_usize(normalizer).unwrap() / b_max.powi(2)
        }
        ObsVecMetric::DTW => {
            // Collect into DVector for the dtw crate
            let x_matrix = DVector::<f32>::from_iterator(
                x.len(),
                x.iter().flat_map(|obs| obs.iter().map(|value| value.as_())),
            );
            let y_matrix = DVector::<f32>::from_iterator(
                y.len(),
                y.iter().flat_map(|obs| obs.iter().map(|value| value.as_())),
            );

            // Compute DTW path
            let dtw_path: Vec<(usize, usize)> = dtw::fast_dtw_with_cmp(
                x_matrix.as_slice(),
                y_matrix.as_slice(),
                x_matrix.len() / 10,
                &dtw::dist::euclidean_distance,
                &f64::total_cmp,
            );

            let dtw_score = dtw_path
                .iter()
                .map(|(i, j)| {
                    let x_vec = &x[*i];
                    let y_vec = &y[*j];

                    if !x_vec.is_valid() && !y_vec.is_valid() {
                        0.0
                    } else if x_vec.is_valid() && y_vec.is_valid() {
                        (x_vec - y_vec).sum_of_squares().as_()
                    } else {
                        f32::INFINITY
                    }
                })
                .sum::<f32>()
                .sqrt()
                / (dtw_path.len() as f32).sqrt();

            T::from_f32(dtw_score).unwrap()
        }
        ObsVecMetric::Valid => {
            let is_valid = x.iter().zip(y).fold(true, |acc, (out_vec, ref_vec)| {
                if (!ref_vec.is_valid() && !out_vec.is_valid())
                    | (ref_vec.is_valid() && out_vec.is_valid())
                {
                    acc & true
                } else {
                    acc & false
                }
            });

            match is_valid {
                true => T::zero(),
                false => T::one() / T::zero(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{DMatrix, Dyn, U1, U5};
    use prodef::{Density, domain::UDomain, multinormal::MultiNormalDensity};

    #[test]
    fn test_obsvec() {
        let ov_1 = ObsVec::<f64, 3>::from([1.0, 2.0, 3.0]);

        assert!(ov_1.sum_of_squares() == 14.0);

        let ov_2 = ObsVec::<f64, 3>::from([2.0, 3.0, 1.0]);

        assert!((ov_1.clone() + ov_2.clone()).sum_of_squares() == 9.0 + 25.0 + 16.0);
        assert!((ov_1.clone() - ov_2.clone()).sum_of_squares() == 6.0);
        assert!(ov_1.clone().mean_square_error(&ov_2) == 2.0);

        assert!((ov_1.clone() * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * ov_1.clone()).sum_of_squares() == 4.0 * 14.0);

        assert!((&ov_1 * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * &ov_1).sum_of_squares() == 4.0 * 14.0);

        assert!((ov_1.clone() / 2.0).sum_of_squares() == 14.0 / 4.0);
        assert!((&ov_1 / 2.0).sum_of_squares() == 14.0 / 4.0);

        assert!(!ObsVec::<f64, 3>::default().is_valid());

        assert!(!ObsVec([f64::NAN, 0.0, 0.0]).is_valid());

        assert!(!ObsVec([0.0, f64::NAN, 0.0, 0.0]).is_valid());
    }

    #[test]
    fn test_ov_metrics() {
        let array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([1.0, 2.0, 3.0])));

        let array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        assert!(ulps_eq!(
            ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::MSE),
            2.0
        ));
        assert!(ulps_eq!(
            ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::RMSE),
            2.0_f64.sqrt()
        ));

        assert!(ulps_eq!(
            ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::NMSE),
            2.0 / 14.0 * 3.0
        ));

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) == 0.0);
    }

    #[test]
    fn test_ov_invalids() {
        let mut array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([1.0, 2.0, 3.0])));

        array_1[0] = ObsVec::default();

        let mut array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) != 0.0);

        array_2[0] = ObsVec::default();

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) == 0.0);

        array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([1.0, 2.0, 3.0])));
        array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObsVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        array_1[4] = ObsVec::default();
        array_2[3] = ObsVec::default();
        array_2[4] = ObsVec::default();

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) != 0.0);

        array_1[0] = ObsVec::default();
        array_2[0] = ObsVec::default();
        array_1[3] = ObsVec::default();

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) == 0.0);

        array_1[2] = ObsVec::default();

        assert!(ov_error(array_1.as_slice(), array_2.as_slice(), ObsVecMetric::Valid) != 0.0);
    }

    #[test]
    fn test_likelihood() {
        let array_1 = DVector::from_iterator(
            7,
            (0..7).map(|idx| {
                if (idx == 0) || (idx == 6) {
                    ObsVec::default()
                } else {
                    ObsVec::<f32, 3>::from([0.1, -0.05, 0.11])
                }
            }),
        );

        let array_2 = DVector::from_iterator(
            7,
            (0..7).map(|idx| {
                if (idx == 0) || (idx == 6) {
                    ObsVec::default()
                } else {
                    ObsVec::<f32, 3>::zeros()
                }
            }),
        );

        let covariance: MultiNormalDensity<_, nalgebra::Dyn, _> = MultiNormalDensity::from_matrix(
            DMatrix::from_diagonal_element(5, 5, 1.0),
            DVector::zeros(5),
            UDomain::new(Dyn(5)),
        )
        .unwrap();

        let mut ll = 1.0;

        for idx in 0..3 {
            let veca = DVector::from_iterator(5, array_1.iter().skip(1).take(5).map(|ov| ov[idx]));
            let vecb = DVector::from_iterator(5, array_2.iter().skip(1).take(5).map(|ov| ov[idx]));

            let delta = &veca - &vecb;

            ll *= covariance.density::<U1, U5>(&delta.as_view()).unwrap();
        }

        assert!(ulps_eq!(ll.ln(), -13.845578));
    }
}
