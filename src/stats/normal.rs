use crate::stats::{Density, DensityRange};
use covmatrix::CovMatrix;
use log::error;
use nalgebra::{Const, Dim, Dyn, MatrixView, RealField, SMatrix, SVector, SVectorView, Scalar, U1};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A multivariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct MultivariateNormalDensity<T, const D: usize>
where
    T: Scalar,
{
    /// A [`CovMatrix`] that describes the underlying multivariate normal probability density function.
    covm: CovMatrix<T, Const<D>>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, D>,

    /// Valid parameter range.
    range: SVector<DensityRange<T>, D>,
}

impl<T, const D: usize> MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    /// Returns a reference to the underlying [`CovMatrix`].
    pub fn covmatrix(&self) -> &CovMatrix<T, Const<D>> {
        &self.covm
    }

    /// Calculates the exact density at a specific position `x`.
    pub fn density(&self, x: &SVectorView<T, D>) -> T
    where
        StandardNormal: Distribution<T>,
    {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covm.pseudo_inverse() * diff)[(0, 0)];

        let p_nonzero: usize = self.covm.matrix().diagonal().iter().fold(0, |acc, next| {
            if next.abs() > T::zero() { acc + 1 } else { acc }
        });

        (-value / T::from_usize(2).unwrap()).exp()
            / ((T::two_pi()).powi(p_nonzero as i32) * self.covm.pseudo_determinant().unwrap())
                .sqrt()
    }

    /// Draw a random sample vector from the underlying density using a custom offset instead of the mean.
    pub fn draw_sample_with_offset<const A: usize>(
        &self,
        offset: &SVectorView<T, D>,
        rng: &mut impl Rng,
    ) -> Option<SVector<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        let normal = StandardNormal;

        let mut proposal = offset
            + self.covm.l().unwrap()
                * SVector::<T, D>::from_iterator((0..D).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view::<_, _, U1, Const<D>>()) {
            proposal = offset
                + self.covm.l().unwrap()
                    * SVector::<T, D>::from_iterator((0..D).map(|_| rng.sample(normal)));

            attempts += 1;

            if attempts > A {
                error!(
                    "MultivariateNormalDensity::draw_sample_with_offset has failed to draw a valid sample after {} tries",
                    attempts
                );

                // easy debug output
                // error!(
                //     "\n\tproposal={}\n\toffset={}\n\t{:?}",
                //     &proposal,
                //     &offset,
                //     self.get_range()
                // );
                // error!("\n\tmatrix={}", self.covm.matrix());

                return None;
            }
        }

        Some(proposal)
    }

    /// Create a [`MultivariateNormalDensity`] from a [`CovMatrix`].
    pub fn from_covmatrix(
        covm: CovMatrix<T, Const<D>>,
        mean: SVector<T, D>,
        range: SVector<DensityRange<T>, D>,
    ) -> Self {
        Self { covm, mean, range }
    }

    /// Create a [`MultivariateNormalDensity`] from an ensemble of particles.
    pub fn from_vectors<'a, RStride: Dim, CStride: Dim>(
        vectors: &MatrixView<T, Const<D>, Dyn, RStride, CStride>,
        range: SVector<DensityRange<T>, D>,
        opt_weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let covm = CovMatrix::from_vectors(vectors, opt_weights, true)?;
        let mut mean = vectors.column_mean();

        // Set mean to first particle value if covariance is zero.
        // This fixes numerical issues where taking the mean over many
        // particles does not equal the constant value.
        covm.matrix()
            .diagonal()
            .iter()
            .zip(mean.iter_mut())
            .enumerate()
            .for_each(|(idx, (cov, value))| {
                if cov.eq(&T::zero()) {
                    *value = vectors[(idx, 0)];
                }
            });

        Some(Self { covm, mean, range })
    }

    /// Generate a [`CovMatrix`] for a kernel density estimator from a multivariate normal density.
    pub fn generate_kde_covmatrix(&self, size: usize) -> CovMatrix<T, Const<D>> {
        let variances = self.covm.matrix().diagonal();

        let d_factor = (T::from_usize(4).unwrap() / T::from_usize(self.covm.rank() + 2).unwrap())
            .powf(T::one() / T::from_usize(self.covm.rank() + 4).unwrap());

        let n_factor = T::one()
            / T::from_usize(size)
                .unwrap()
                .powf(T::one() / T::from_usize(self.covm.rank() + 4).unwrap());

        let diagonal = SVector::<T, D>::from_iterator(
            variances
                .iter()
                .map(|variance| (d_factor * n_factor).powi(2) * *variance),
        );

        CovMatrix::new(SMatrix::<T, D, D>::from_diagonal(&diagonal), true).unwrap()
    }

    /// Compute the Kullback-Leibler divergence between two [`MultivariateNormalDensity`].
    pub fn kullback_leibler_divergence(&self, other: &MultivariateNormalDensity<T, D>) -> Option<T>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let mut l_0 = *self.covm.l().unwrap();
        let mu_0 = &self.mean;

        let mut l_1 = *other.covm.l().unwrap();
        let mu_1 = &other.mean;

        let mut p_nonzero = D;

        // Detect zero'd columns/rows that need to be modified.
        (0..l_0.nrows()).for_each(|idx| {
            if l_0[(idx, idx)].eq(&T::zero()) {
                l_0[(idx, idx)] = T::one() / T::zero();

                p_nonzero -= 1;

                // Set off diagonals to zero.
                for jdx in 0..l_0.ncols() {
                    if jdx != idx {
                        l_0[(idx, jdx)] = T::zero();
                        l_0[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        let m = l_0.clone().solve_lower_triangular(&l_1).unwrap();

        // Detect zero'd columns/rows that need to be modified.
        (0..l_1.nrows()).for_each(|idx| {
            if l_1[(idx, idx)].eq(&T::zero()) {
                l_1[(idx, idx)] = T::one() / T::zero();

                // Set off diagonals to zero.
                for jdx in 0..l_1.ncols() {
                    if jdx != idx {
                        l_1[(idx, jdx)] = T::zero();
                        l_1[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        let y = l_1.clone().solve_lower_triangular(&(mu_1 - mu_0)).unwrap();

        Some(
            (m.iter().sum::<T>() - T::from_usize(p_nonzero).unwrap()
                + y.norm()
                + T::from_usize(2).unwrap()
                    * l_1
                        .diagonal()
                        .iter()
                        .zip(l_0.diagonal().iter())
                        .map(|(a, b)| {
                            if a.is_finite() && b.is_finite() {
                                (*a / *b).ln()
                            } else {
                                T::zero()
                            }
                        })
                        .sum::<T>())
                / T::from_usize(2).unwrap(),
        )
    }

    /// Returns the center of the multivariate normal density.
    pub fn mean(&self) -> &SVector<T, D> {
        &self.mean
    }

    /// Returns a reference to the inverse of the underlying covariance matrix.
    pub fn pseudo_inverse(&self) -> &SMatrix<T, D, D> {
        self.covm.pseudo_inverse()
    }
}

impl<T, const D: usize> Density<T, D> for &MultivariateNormalDensity<T, D>
where
    T: Copy + RealField,
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        let normal = StandardNormal;

        let mut proposal = self.mean
            + self.covm.l().unwrap()
                * SVector::<T, D>::from_iterator((0..D).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view::<_, _, U1, Const<D>>()) {
            proposal = self.mean
                + self.covm.l().unwrap()
                    * SVector::<T, D>::from_iterator((0..D).map(|_| rng.sample(normal)));

            attempts += 1;

            if attempts > A {
                error!(
                    "MultivariateNormalDensity::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );

                return None;
            }
        }

        Some(proposal)
    }

    fn get_constants(&self) -> SVector<T, D> {
        SVector::from_iterator(
            self.covm
                .matrix()
                .diagonal()
                .iter()
                .zip(self.mean.iter())
                .map(|(cov, mean)| {
                    if cov.eq(&T::zero()) {
                        *mean
                    } else {
                        (-T::one()).sqrt()
                    }
                }),
        )
    }

    fn get_range(&self) -> SVector<DensityRange<T>, D> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, D>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let diff = x - self.mean;
        let value = (diff.transpose() * self.covm.pseudo_inverse() * diff)[(0, 0)];

        (-value / T::from_usize(2).unwrap()).exp()
    }
}

impl<T, const D: usize> Add<&SVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn add(self, rhs: &SVector<T, D>) -> Self::Output {
        Self {
            covm: self.covm,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, const D: usize> AddAssign<&SVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField,
{
    fn add_assign(&mut self, rhs: &SVector<T, D>) {
        self.mean += rhs
    }
}

impl<T, const D: usize> Mul<T> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covm: self.covm * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<T, const D: usize> MulAssign<T> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covm *= rhs;
    }
}

impl<T, const D: usize> Sub<&SVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn sub(self, rhs: &SVector<T, D>) -> Self::Output {
        Self {
            covm: self.covm,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, const D: usize> SubAssign<&SVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &SVector<T, D>) {
        self.mean -= rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Matrix, SVector, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_multivariate_normal_density() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let covm = CovMatrix::from_vectors::<Dyn, U3>(&array.as_view(), None, true).unwrap();

        let mvpdf = &MultivariateNormalDensity::from_covmatrix(
            covm,
            SVector::from([0.1, 0.0, 0.25]),
            SVector::from([DensityRange::new((-0.75, 0.75)); 3]),
        );

        assert!(ulps_eq!(
            mvpdf.density(&SVector::from([0.2, 0.0, 0.35]).as_view()),
            0.1611928
        ));

        assert!(
            mvpdf
                .relative_density(&SVector::from([0.2f32, 0.1, 0.35]).as_view())
                .is_nan()
        );

        assert!(ulps_eq!(
            mvpdf.draw_sample::<100>(&mut rng).unwrap(),
            SVector::from([-0.4150916, 0.0, 0.4898513])
        ));

        assert!(mvpdf.validate_sample(&mvpdf.draw_sample::<100>(&mut rng).unwrap().as_view()));

        let mvpdf_ensbl = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array.as_view(),
            SVector::from([DensityRange::new((-0.75, 0.75)); 3]),
            None,
        )
        .unwrap();

        assert!(ulps_eq!(
            mvpdf.covm.l().unwrap(),
            mvpdf_ensbl.covm.l().unwrap()
        ));
    }

    #[test]
    fn test_multivariate_normal_density_kld() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array_1 = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let array_2 = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    0.25 + rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let mvpdf_1 = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array_1.as_view(),
            SVector::from([DensityRange::new((-2.0, 2.0)); 3]),
            None,
        )
        .unwrap();
        let mvpdf_2 = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array_2.as_view(),
            SVector::from([DensityRange::new((-2.0, 2.0)); 3]),
            None,
        )
        .unwrap();

        assert!(ulps_eq!(
            mvpdf_1.kullback_leibler_divergence(&mvpdf_2).unwrap(),
            0.20391017
        ));
    }
}
