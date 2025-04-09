use crate::{
    stats::{CovMatrix, PDF, PDFExactDensity, StatsError},
    t_from,
};
use log::warn;
use nalgebra::{Const, DMatrix, Dyn, MatrixView, RealField, SVector, SVectorView, Scalar};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A multivariate normal PDF .
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFMultivariate<T, const P: usize>
where
    T: Copy + Scalar,
{
    /// A [`CovMatrix<T>`] that describes the underlying multivariate normal PDF.
    covmat: CovMatrix<T>,

    /// The mean parameter of the multivariate normal distribution.
    mean: SVector<T, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(T, T); P],
}

impl<T, const P: usize> PDFMultivariate<T, P>
where
    T: Copy + Float + RealField + Scalar + Sum<T> + for<'x> Sum<&'x T>,
{
    /// Returns a reference to the lower triangular matrix L from the Cholesky decomposition
    /// of the covariance matrix.
    pub fn ref_cholesky_ltm(&self) -> &DMatrix<T> {
        &self.covmat.ref_cholesky_ltm()
    }

    /// Create a [`PDFMultivariate`] from a covariance matrix.
    pub fn from_covmat(covmat: CovMatrix<T>, mean: SVector<T, P>, range: [(T, T); P]) -> Self {
        Self {
            covmat,
            mean,
            range,
        }
    }

    /// Create a [`PDFMultivariate`] from an ensemble of particles.
    pub fn from_particles<'a>(
        particles: &MatrixView<'a, T, Const<P>, Dyn>,
        range: [(T, T); P],
        weights: Option<&[T]>,
    ) -> Result<Self, StatsError<T>> {
        let covmat = CovMatrix::from_particles(&particles.as_view(), weights)?;

        Ok(Self {
            covmat,
            mean: particles.column_mean(),
            range,
        })
    }

    /// Returns a reference to the inverse of the underlying covariance matrix.
    pub fn ref_inverse_matrix(&self) -> &DMatrix<T> {
        &self.covmat.ref_inverse_matrix()
    }

    /// Compute the Kullback-Leibler divergence between two [`PDFMultivariate`].
    pub fn kullback_leibler_divergence(&self, other: &PDFMultivariate<T, P>) -> T {
        let l_0 = self.covmat.ref_cholesky_ltm();
        let mu_0 = self.mean;

        let l_1 = other.covmat.ref_cholesky_ltm();
        let mu_1 = other.mean;

        let m = l_0.clone().cholesky().unwrap().solve(&l_1);
        let y = l_1.clone().cholesky().unwrap().solve(&(&mu_1 - &mu_0));

        t_from!(0.5)
            * (m.iter().sum::<T>() - T::from_usize(P).unwrap()
                + y.norm()
                + t_from!(2.0)
                    * l_1
                        .diagonal()
                        .iter()
                        .zip(l_0.diagonal().iter())
                        .map(|(a, b)| Float::ln(*a / *b))
                        .sum::<T>())
    }

    /// Returns a reference to the underlying covariance matrix.
    pub fn ref_matrix(&self) -> &DMatrix<T> {
        &self.covmat.ref_matrix()
    }
}

impl<T, const P: usize> PDF<T, P> for &PDFMultivariate<T, P>
where
    T: Copy + Float + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_inverse_matrix() * diff)[(0, 0)];

        Float::exp(t_from!(-0.5) * value)
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let normal = Normal::new(T::zero(), T::one()).unwrap();

        let mut proposal = self.mean
            + self.covmat.ref_cholesky_ltm()
                * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = self.mean
                + self.covmat.ref_cholesky_ltm()
                    * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 150) && (attempts % 150 == 0) {
                warn!(
                    "PDFMultivariate::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(T, T); P] {
        self.range
    }
}

impl<T, const P: usize> PDFExactDensity<T, P> for &PDFMultivariate<T, P>
where
    T: Copy + Float + RealField + Scalar,
    StandardNormal: Distribution<T>,
{
    fn exact_density(&self, x: &SVectorView<T, P>) -> T {
        let diff = x - self.mean;
        let value = (diff.transpose() * self.covmat.ref_inverse_matrix() * diff)[(0, 0)];

        Float::exp(t_from!(-0.5) * value)
            / Float::sqrt(Float::powi(t_from!(2.0) * T::pi(), P as i32) * self.covmat.determinant())
    }
}

impl<T, const P: usize> Add<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn add(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean + rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> AddAssign<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn add_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean += rhs
    }
}

impl<T, const P: usize> Mul<T> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            mean: self.mean,
            range: self.range,
        }
    }
}

impl<const P: usize> Mul<PDFMultivariate<f32, P>> for f32 {
    type Output = PDFMultivariate<f32, P>;

    fn mul(self, rhs: PDFMultivariate<f32, P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<const P: usize> Mul<PDFMultivariate<f64, P>> for f64 {
    type Output = PDFMultivariate<f64, P>;

    fn mul(self, rhs: PDFMultivariate<f64, P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            mean: rhs.mean,
            range: rhs.range,
        }
    }
}

impl<T, const P: usize> MulAssign<T> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covmat *= rhs;
    }
}

impl<T, const P: usize> Sub<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFMultivariate<T, P>;

    fn sub(self, rhs: &SVector<T, P>) -> Self::Output {
        Self {
            covmat: self.covmat,
            mean: self.mean - rhs,
            range: self.range,
        }
    }
}

impl<T, const P: usize> SubAssign<&SVector<T, P>> for PDFMultivariate<T, P>
where
    T: Copy + RealField + Scalar,
{
    fn sub_assign(&mut self, rhs: &SVector<T, P>) {
        self.mean -= rhs
    }
}
