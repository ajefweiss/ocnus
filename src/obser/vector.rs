use crate::base::ScObsSeries;
use crate::obser::{OcnusNoise, OcnusObser};
use covmatrix::CovMatrix;
use derive_more::{Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use nalgebra::{Const, DVector, DVectorView, Dyn, OVector, RealField, SVector};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::ops::Div;
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
    [T; N],
);

impl<T, const N: usize> ObserVec<T, N>
where
    T: Copy + RealField,
{
    /// Returns true if any entry within the observation vector is `NaN`.
    pub fn any_nan(&self) -> bool {
        !self.iter().fold(true, |acc, next| acc & next.is_finite())
    }

    /// Returns true if all entries within the observation vector are `NaN`.
    pub fn is_nan(&self) -> bool {
        self.iter().fold(true, |acc, next| acc & !next.is_finite())
    }

    /// Calculate the mean square error between two observation vectors.
    ///
    /// If both observations are not considered valid, returns `0.0`.
    /// If only one of the two observations is considered valid, returns `NaN`.
    pub fn mean_square_error(&self, other: &Self) -> T
    where
        T: Sum,
        Self: OcnusObser,
    {
        if self.is_valid() & other.is_valid() {
            (self - other).sum_of_squares() / T::from_usize(N).unwrap()
        } else if !self.is_valid() & !other.is_valid() {
            T::zero()
        } else {
            (-T::one()).sqrt()
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

impl<T> Display for ObserVec<T, 12>
where
    T: RealField,
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

impl<T, const N: usize> OcnusObser for ObserVec<T, N>
where
    T: Copy + RealField,
{
    fn is_valid(&self) -> bool {
        !self.any_nan()
    }
}

impl<T, const N: usize> PartialEq for ObserVec<T, N>
where
    T: RealField,
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
pub enum ObserVecNoise<T, const N: usize>
where
    T: RealField,
{
    Gaussian(T, u64),
    Multivariate(CovMatrix<T, Const<N>>, u64),
}

impl<T, const N: usize> OcnusNoise<T, ObserVec<T, N>> for ObserVecNoise<T, N>
where
    T: Copy + RealField,
    StandardNormal: Distribution<T>,
{
    fn generate_noise(
        &self,
        series: &ScObsSeries<T>,
        rng: &mut impl rand::Rng,
    ) -> nalgebra::DVector<ObserVec<T, N>> {
        match self {
            ObserVecNoise::Gaussian(std_dev, ..) => {
                let normal = StandardNormal;
                let size = series.len();

                DVector::from_iterator(
                    size,
                    (0..size).map(|_| ObserVec([rng.sample(normal) * *std_dev; N])),
                )
            }
            ObserVecNoise::Multivariate(covmat, ..) => {
                let normal = StandardNormal;
                let size = series.len();

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

/// Compute the multivariate likelihood for an observation `x` with expected mean `mu` and covariance matrix `covm`.
pub fn multivariate_log_likelihood<T, const N: usize>(
    x: &DVectorView<ObserVec<T, N>>,
    mu: &DVectorView<ObserVec<T, N>>,
    covm: &CovMatrix<T, Dyn>,
) -> T
where
    T: Copy + RealField,
    usize: AsPrimitive<T>,
{
    let mut value = T::zero();

    // Correct for double NaN's. The covariance matrix `covm` must have the correct dimensions for after NaN removal.
    let count_x = x
        .iter()
        .fold(0, |acc, next| if !next.is_nan() { acc + 1 } else { acc });

    let count_mu = mu
        .iter()
        .fold(0, |acc, next| if !next.is_nan() { acc + 1 } else { acc });

    if count_x != count_mu {
        return -T::one() / T::zero();
    }

    // Extra case where the NaN's are not in the same place.
    let valid: bool = x.iter().zip(mu.iter()).fold(true, |acc, next| {
        if !(next.0.is_nan() ^ next.1.is_nan()) {
            acc
        } else {
            false
        }
    });

    if !valid {
        return -T::one() / T::zero();
    }

    for i in 0..N {
        let x_slice = DVector::<T>::from_iterator(
            count_x,
            x.iter().zip(mu.iter()).filter_map(|(x_obs, mu_obs)| {
                if !x_obs.is_nan() && !mu_obs.is_nan() {
                    Some(x_obs[i])
                } else {
                    None
                }
            }),
        );
        let mu_slice = DVector::<T>::from_iterator(
            count_mu,
            mu.iter().zip(x.iter()).filter_map(|(mu_obs, x_obs)| {
                if !mu_obs.is_nan() && !x_obs.is_nan() {
                    Some(mu_obs[i])
                } else {
                    None
                }
            }),
        );

        value += covm.multivariate_log_likelihood(&x_slice.as_view(), &mu_slice.as_view())
    }

    value
}

/// Mean square error (MSE) for the [`ObserVec`] type.
pub fn observec_mse<T, const N: usize>(
    obser: &DVectorView<ObserVec<T, N>>,
    other: &DVectorView<ObserVec<T, N>>,
) -> T
where
    T: Copy + RealField + Sum,
{
    // Correct for double NaN's
    let normalizer = obser.fold(0, |acc, next| if !next.is_nan() { acc + 1 } else { acc });

    obser
        .into_iter()
        .zip(other)
        .map(|(out_vec, ref_vec)| {
            if ref_vec.is_nan() && out_vec.is_nan() {
                T::zero()
            } else if !ref_vec.is_nan() && !out_vec.is_nan() {
                (ref_vec - out_vec).sum_of_squares()
            } else {
                T::one() / T::zero()
            }
        })
        .sum::<T>()
        / T::from_usize(normalizer * N).unwrap()
}

/// Mean square error (MSE) percentage for the [`ObserVec`] type.
pub fn observec_msep<T, const N: usize>(
    obser: &DVectorView<ObserVec<T, N>>,
    other: &DVectorView<ObserVec<T, N>>,
) -> T
where
    T: Copy + RealField + Sum,
{
    // Correct for double NaN's
    let normalize_vector = DVector::from_iterator(
        obser.len(),
        obser.iter().map(|obs| {
            if obs.any_nan() {
                ObserVec::default()
            } else {
                ObserVec::zeros()
            }
        }),
    );

    observec_mse(obser, other) / observec_mse(&normalize_vector.as_view(), other)
}

/// Root mean square error (RMSE) for the [`ObserVec`] type.
pub fn observec_rmse<T, const N: usize>(
    obser: &DVectorView<ObserVec<T, N>>,
    other: &DVectorView<ObserVec<T, N>>,
) -> T
where
    T: Copy + RealField + Sum,
{
    observec_mse(obser, other).sqrt()
}

/// Root mean square error (RMSE) percentage for the [`ObserVec`] type.
pub fn observec_rmsep<T, const N: usize>(
    obser: &DVectorView<ObserVec<T, N>>,
    other: &DVectorView<ObserVec<T, N>>,
) -> T
where
    T: Copy + RealField + Sum,
{
    let normalize_vector = DVector::from_iterator(
        obser.len(),
        obser.iter().map(|obs| {
            if obs.any_nan() {
                ObserVec::default()
            } else {
                ObserVec::zeros()
            }
        }),
    );

    (observec_mse(obser, other) / observec_mse(&normalize_vector.as_view(), other)).sqrt()
}

#[cfg(test)]
mod tests {
    use approx::ulps_eq;
    use nalgebra::DMatrix;

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

        assert!((ov_1.clone() / 2.0).sum_of_squares() == 14.0 / 4.0);
        assert!((&ov_1 / 2.0).sum_of_squares() == 14.0 / 4.0);

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

    #[test]
    fn test_observec_metrics() {
        let array_1 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([1.0, 2.0, 3.0])));

        let array_2 =
            DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f64, 3>::from([2.0, 3.0, 1.0])));

        assert!(observec_mse(&array_1.as_view(), &array_2.as_view()) == 2.0);
        assert!(observec_rmse(&array_1.as_view(), &array_2.as_view()) == 2.0_f64.sqrt());

        assert!(observec_msep(&array_1.as_view(), &array_2.as_view()) == 2.0 / 56.0);
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

        let array_1s = DVector::from_iterator(
            5,
            (0..5).map(|_| ObserVec::<f32, 3>::from([0.1, -0.05, 0.11])),
        );

        let array_2s = DVector::from_iterator(5, (0..5).map(|_| ObserVec::<f32, 3>::zeros()));

        let covm = CovMatrix::new(DMatrix::from_diagonal_element(5, 5, 1.0), true).unwrap();

        let ll = multivariate_log_likelihood(&array_1.as_view(), &array_2.as_view(), &covm);
        let lls = multivariate_log_likelihood(&array_1s.as_view(), &array_2s.as_view(), &covm);

        assert!(ulps_eq!(ll, -47.18538));
        assert!(ulps_eq!(ll, lls));
    }
}
