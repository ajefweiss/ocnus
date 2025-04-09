use crate::{
    stats::{PDF, StatsError},
    t_from,
};
use core::panic;
use log::warn;
use nalgebra::{
    Const, Dyn, Matrix, MatrixView, RealField, SVector, SVectorView, Scalar, ViewStorage,
};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    cmp::Ordering,
    iter::Sum,
    ops::{Mul, MulAssign},
};

use super::PDFMultivariate;

/// A PDF defined by references to an ensemble of particles.
#[derive(Debug)]
pub struct PDFParticles<'a, T, const P: usize>
where
    T: Copy + Scalar,
{
    /// A [`PDFMultivariate`] that describes an estimate of the underlying PDF.
    mvpdf: PDFMultivariate<T, P>,

    /// The particle ensemble that describes the underlying PDF.
    particles: MatrixView<'a, T, Const<P>, Dyn>,

    /// Valid parameter range.
    range: [(T, T); P],

    /// Particle ensemble weights
    weights: Vec<T>,
}

impl<'a, T, const P: usize> PDFParticles<'a, T, P>
where
    T: Copy + Float + RealField + SampleUniform + Scalar + Sum<T> + for<'x> Sum<&'x T>,
{
    /// Create a new [`PDFParticles`] from a particle matrix view.
    pub fn from_particles(
        particles: MatrixView<'a, T, Const<P>, Dyn>,
        range: [(T, T); P],
        weights: Vec<T>,
    ) -> Result<Self, StatsError<T>> {
        let mvpdf =
            PDFMultivariate::from_particles(&particles.as_view(), range, Some(weights.as_slice()))?;

        Ok(Self {
            mvpdf,
            particles,
            range,
            weights,
        })
    }

    /// Create a new [`PDFParticles`] with importance weights for a set of particles assuming a transition from `ptpdf_from`.
    /// NOTE: This implementation is slower than previous versions for some reason (factor 2?!?).
    pub fn from_particles_and_ptpdf<D>(
        particles: Matrix<T, Const<P>, Dyn, ViewStorage<'a, T, Const<P>, Dyn, Const<1>, Const<P>>>,
        ptpdf_from: &PDFParticles<T, P>,
        prior: &D,
    ) -> Result<PDFParticles<'a, T, P>, StatsError<T>>
    where
        T: Copy + Float + RealField + SampleUniform + Sum<T> + for<'x> Sum<&'x T>,
        D: PDF<T, P>,
    {
        let covmat_inv = ptpdf_from.mvpdf.ref_inverse_matrix();

        let mut weights = particles
            .par_column_iter()
            .map(|params_new| {
                let value = ptpdf_from
                    .particles
                    .column_iter()
                    .zip(ptpdf_from.weights.iter())
                    .map(|(params_old, weight_old)| {
                        let delta = params_new - params_old;

                        Float::exp(
                            Float::ln(*weight_old)
                                - (delta.transpose() * covmat_inv * delta)[(0, 0)],
                        )
                    })
                    .sum::<T>();

                (T::one() / value) * prior.relative_density(&params_new)
            })
            .collect::<Vec<T>>();

        let weights_total = weights.iter().sum::<T>();

        weights
            .iter_mut()
            .for_each(|weight| *weight /= weights_total);

        PDFParticles::from_particles(particles, ptpdf_from.range, weights)
    }

    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Estimates the Kullback-Leibler divergence between two [`PDFParticles`]
    /// assuming the underlying PDF is multivariate (i.e. a [`PDFMultivariate`](crate::stats::PDFMultivariate)).
    pub fn kullback_leibler_divergence_mvpdf_estimate(&self, other: &PDFParticles<T, P>) -> T
where {
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
                *value = T::infinity();
            }
        });

        let m = l_1.solve_lower_triangular(&l_0).unwrap();
        let y = l_1.solve_lower_triangular(&(&mu_1 - &mu_0)).unwrap();

        l_1.iter_mut().step_by(P + 1).for_each(|value| {
            if *value == T::infinity() {
                *value = T::zero();
            }
        });

        let m_sum = m.iter().map(|value| *value * *value).sum::<T>();
        let ln_sum = t_from!(2.0)
            * l_1
                .diagonal()
                .iter()
                .zip(l_0.diagonal().iter())
                .map(|(a, b)| {
                    if (a.partial_cmp(&T::zero()).unwrap() == Ordering::Greater)
                        && (b.partial_cmp(&T::zero()).unwrap() == Ordering::Greater)
                    {
                        Float::ln(*a / *b)
                    } else {
                        T::zero()
                    }
                })
                .sum::<T>();

        t_from!(0.5) * (m_sum - k + Float::powi(y.norm(), 2) + ln_sum)
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
    pub fn resample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
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

        Ok(offset.into())
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &Vec<T> {
        &self.weights
    }
}

impl<T, const P: usize> PDF<T, P> for &PDFParticles<'_, T, P>
where
    T: Float + RealField + SampleUniform + Scalar + Sum<T> + for<'x> Sum<&'x T>,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, _x: &SVectorView<T, P>) -> T {
        unimplemented!()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let normal = Normal::new(T::zero(), T::one()).unwrap();
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

            if (attempts > 5000) && (attempts % 5000 == 0) {
                warn!(
                    "PDFParticles::draw_sample has failed to draw a valid sample after {} tries",
                    attempts,
                );

                panic!()
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(T, T); P] {
        self.range
    }
}

impl<'a, T, const P: usize> Mul<T> for PDFParticles<'a, T, P>
where
    T: Copy + RealField + Scalar,
{
    type Output = PDFParticles<'a, T, P>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            mvpdf: self.mvpdf * rhs,
            particles: self.particles,
            range: self.range,
            weights: self.weights,
        }
    }
}

impl<'a, const P: usize> Mul<PDFParticles<'a, f32, P>> for f32 {
    type Output = PDFParticles<'a, f32, P>;

    fn mul(self, rhs: PDFParticles<'a, f32, P>) -> Self::Output {
        Self::Output {
            mvpdf: self * rhs.mvpdf,
            particles: rhs.particles,
            range: rhs.range,
            weights: rhs.weights,
        }
    }
}

impl<'a, const P: usize> Mul<PDFParticles<'a, f64, P>> for f64 {
    type Output = PDFParticles<'a, f64, P>;

    fn mul(self, rhs: PDFParticles<'a, f64, P>) -> Self::Output {
        Self::Output {
            mvpdf: self * rhs.mvpdf,
            particles: rhs.particles,
            range: rhs.range,
            weights: rhs.weights,
        }
    }
}

impl<T, const P: usize> MulAssign<T> for PDFParticles<'_, T, P>
where
    T: Copy + RealField,
{
    fn mul_assign(&mut self, rhs: T) {
        self.mvpdf *= rhs;
    }
}
