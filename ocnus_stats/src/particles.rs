use log::error;
use nalgebra::{
    Const, Dyn, Matrix, MatrixView, RealField, SMatrix, SVector, SVectorView, Scalar, ViewStorage,
};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Mul, MulAssign},
};

use crate::{CovMatrix, OcnusPDF};

use super::{MultivariateND, StatsError};

/// A PDF defined by an ensemble of particles.
#[derive(Debug)]
pub struct ParticlesND<'a, T, const P: usize>
where
    T: Clone + Scalar,
{
    /// Bandwidth covariance matrix for the kernel density estimation.
    bandwidth: Option<CovMatrix<T>>,

    /// A [`MultivariateND`] that describes an estimate of the underlying PDF.
    mvpdf: MultivariateND<T, P>,

    /// The particle ensemble that approximates the underlying PDF.
    particles: &'a MatrixView<'a, T, Const<P>, Dyn>,

    /// Valid parameter range.
    range: [(T, T); P],

    /// Particle ensemble weights
    weights: Vec<T>,
}

impl<'a, T, const P: usize> ParticlesND<'a, T, P>
where
    T: Copy + RealField + SampleUniform + Scalar,
{
    /// Create a new [`ParticlesND`] from a particle matrix view.
    pub fn from_particles(
        particles: &'a MatrixView<'a, T, Const<P>, Dyn>,
        range: [(T, T); P],
        weights: Vec<T>,
    ) -> Option<Self>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        let mvpdf =
            MultivariateND::from_particles(&particles.as_view(), range, Some(weights.as_slice()))?;

        Some(Self {
            bandwidth: None,
            mvpdf,
            particles,
            range,
            weights,
        })
    }

    /// Create a new [`ParticlesND`] with importance weights for a set of particles assuming a transition from `ptpdf_from`.
    /// NOTE: This implementation is slower than previous versions for some reason (factor 2?!?).
    pub fn from_particles_and_ptpdf<D>(
        particles: &'a Matrix<
            T,
            Const<P>,
            Dyn,
            ViewStorage<'a, T, Const<P>, Dyn, Const<1>, Const<P>>,
        >,
        ptpdf_from: &ParticlesND<T, P>,
        prior: &D,
    ) -> Option<ParticlesND<'a, T, P>>
    where
        T: Copy + RealField + SampleUniform + Sum<T> + for<'x> Sum<&'x T>,
        D: OcnusPDF<T, P>,
    {
        let covmat_inv = ptpdf_from.mvpdf.ref_matrix_inverse();

        let mut weights = particles
            .par_column_iter()
            .map(|params_new| {
                let value = ptpdf_from
                    .particles
                    .column_iter()
                    .zip(ptpdf_from.weights.iter())
                    .map(|(params_old, weight_old)| {
                        let delta = params_new - params_old;

                        ((*weight_old).ln() - (delta.transpose() * covmat_inv * delta)[(0, 0)])
                            .exp()
                    })
                    .sum::<T>();

                (T::one() / value) * prior.density_rel(&params_new)
            })
            .collect::<Vec<T>>();

        let weights_total = weights.iter().sum::<T>();

        weights
            .iter_mut()
            .for_each(|weight| *weight /= weights_total);

        ParticlesND::from_particles(particles, ptpdf_from.range, weights)
    }

    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Estimates the Kullback-Leibler divergence between two [`ParticlesND`]
    /// using the approximated multivariate normal distributions.
    pub fn kullback_leibler_divergence_mvpdf_estimate(&self, other: &ParticlesND<T, P>) -> T
    where
        T: Sum,
    {
        let l_0 = self.mvpdf.ref_cholesky_ltm();
        let mu_0 = self.particles.column_mean();

        let mut l_1 = other.mvpdf.ref_cholesky_ltm().clone();
        let mu_1 = other.particles.column_mean();

        let k = l_1
            .diagonal()
            .iter()
            .map(|value| {
                if *value == T::zero() {
                    T::zero()
                } else {
                    T::one()
                }
            })
            .sum::<T>();

        // Modify zero entries in l_1 to one to allow for solving of equation systems.
        l_1.iter_mut().step_by(P + 1).for_each(|value| {
            if *value == T::zero() {
                *value = T::one() / T::zero();
            }
        });

        let m = l_1.solve_lower_triangular(&l_0).unwrap();
        let y = l_1.solve_lower_triangular(&(&mu_1 - &mu_0)).unwrap();

        l_1.iter_mut().step_by(P + 1).for_each(|value| {
            if *value == T::one() / T::zero() {
                *value = T::zero();
            }
        });

        let m_sum = m.iter().map(|value| *value * *value).sum::<T>();
        let ln_sum = T::from_usize(2).unwrap()
            * l_1
                .diagonal()
                .iter()
                .zip(l_0.diagonal().iter())
                .map(|(a, b)| {
                    if (a.partial_cmp(&T::zero()).unwrap() == Ordering::Greater)
                        && (b.partial_cmp(&T::zero()).unwrap() == Ordering::Greater)
                    {
                        (*a / *b).ln()
                    } else {
                        T::zero()
                    }
                })
                .sum::<T>();

        (m_sum - k + (y.norm()).powi(2) + ln_sum) / T::from_usize(2).unwrap()
    }

    /// Returns the number of particles in the ensemble.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Access the ensemble particle matrix.
    pub fn particles_ref(&self) -> MatrixView<T, Const<P>, Dyn> {
        self.particles.as_view()
    }

    /// Resample from existing particles.
    pub fn resample(&self, rng: &mut impl Rng) -> SVector<T, P> {
        let uniform = Uniform::new(T::zero(), T::one()).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx: T = rng.sample(uniform);

                // Here we abuse try_fold to return particle index early wrapped within Err().
                match self
                    .weights
                    .iter()
                    .enumerate()
                    .try_fold(T::zero(), |acc, (idx, weight)| {
                        let next_weight = acc + *weight;
                        if wdx < next_weight {
                            Err(idx)
                        } else {
                            Ok(next_weight)
                        }
                    }) {
                    Ok(_) => self.weights.len() - 1,
                    Err(idx) => idx,
                }
            };

            self.particles.column(pdx)
        };

        offset.into()
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &Vec<T> {
        &self.weights
    }
}

impl<T, const P: usize> OcnusPDF<T, P> for &ParticlesND<'_, T, P>
where
    T: Copy + RealField + SampleUniform + Sum,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &SVectorView<T, P>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let hmat =
            match &self.bandwidth {
                Some(covmatrix) => covmatrix,
                None => {
                    let variances = self.mvpdf.ref_matrix().diagonal();

                    let p_nonzero: usize = variances.iter().fold(0, |acc, next| {
                        if next.abs() > T::zero() { acc + 1 } else { acc }
                    });

                    let d_factor = (T::from_usize(4).unwrap()
                        / T::from_usize(p_nonzero + 2).unwrap())
                    .powf(T::one() / T::from_usize(p_nonzero + 4).unwrap());

                    let n_factor = T::one()
                        / T::from_usize(self.particles.ncols())
                            .unwrap()
                            .powf(T::one() / T::from_usize(p_nonzero + 4).unwrap());

                    let diagonal = SVector::<T, P>::from_iterator(
                        variances
                            .iter()
                            .map(|variance| (d_factor * n_factor).powi(2) * *variance),
                    );

                    let matrix = SMatrix::<T, P, P>::from_diagonal(&diagonal);

                    &CovMatrix::from_matrix(&matrix.as_view()).unwrap()
                }
            };

        let p_nonzero: usize = hmat.ref_matrix().diagonal().iter().fold(0, |acc, next| {
            if next.abs() > T::zero() { acc + 1 } else { acc }
        });

        let normalization = T::one()
            / T::two_pi().powf(T::from_usize(p_nonzero).unwrap() / T::from_usize(2).unwrap())
            / hmat.determinant().sqrt();

        normalization
            * self
                .particles
                .column_iter()
                .zip(self.weights.iter())
                .map(|(col, weight)| {
                    *weight
                        * (-((x - col).transpose() * hmat.ref_matrix_inverse() * (x - col))[(0, 0)]
                            / T::from_usize(2).unwrap())
                        .exp()
                })
                .sum::<T>()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let normal = StandardNormal;
        let uniform = Uniform::new(T::zero(), T::one()).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx = rng.sample(uniform);

                // Here we abuse try_fold to return particle index early wrapped within Err().
                match self
                    .weights
                    .iter()
                    .enumerate()
                    .try_fold(T::zero(), |acc, (idx, weight)| {
                        let next_weight = acc + *weight;
                        if wdx < next_weight {
                            Err(idx)
                        } else {
                            Ok(next_weight)
                        }
                    }) {
                    Ok(_) => self.weights.len() - 1,
                    Err(idx) => idx,
                }
            };

            self.particles.column(pdx)
        };

        let mut proposal = offset
            + self.mvpdf.ref_cholesky_ltm()
                * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = offset
                + self.mvpdf.ref_cholesky_ltm()
                    * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if attempts > 1999 {
                error!(
                    "ParticlesND::draw_sample has failed to draw a valid sample after {} tries",
                    attempts,
                );

                return Err(StatsError::InefficientSampling {
                    name: "ParticleND",
                    count: 2000,
                });
            }
        }

        Ok(proposal)
    }

    fn get_valid_range(&self) -> [(T, T); P] {
        self.range
    }
}

impl<'a, T, const P: usize> Mul<T> for ParticlesND<'a, T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = ParticlesND<'a, T, P>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            bandwidth: None,
            mvpdf: self.mvpdf * rhs,
            particles: self.particles,
            range: self.range,
            weights: self.weights,
        }
    }
}

impl<T, const P: usize> MulAssign<T> for ParticlesND<'_, T, P>
where
    T: Copy + RealField + Scalar,
{
    fn mul_assign(&mut self, rhs: T) {
        self.mvpdf *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_particlend() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    (idx % 3) as f32 + rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let array_view = &array.as_view();

        let ptpdf =
            ParticlesND::from_particles(array_view, [(-0.75, 0.75); 3], vec![1.0 / 10000.0; 10000])
                .unwrap();

        assert!(
            ((&ptpdf).density_rel(&SVector::from([0.2, -0.15, 0.35]).as_view()) - 0.041557457)
                .abs()
                < 1e-6
        );

        assert!(
            (&ptpdf)
                .density_rel(&SVector::from([2.2, 0.0, 0.35]).as_view())
                .is_nan()
        );

        assert!(
            ((&ptpdf).draw_sample(&mut rng).unwrap()
                - SVector::from([0.44691825, 0.0, 0.38601732,]))
            .norm()
                < 1e-6
        );

        assert!((&ptpdf).validate_sample(&(&ptpdf).draw_sample(&mut rng).unwrap().as_view()));
    }
}
