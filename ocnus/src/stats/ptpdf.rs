use crate::stats::{CovMatrix, PDF, StatsError};
use core::panic;
use log::warn;
use nalgebra::{Const, Dyn, MatrixView, RealField, SVector, SVectorView, Scalar};
use num_traits::Float;
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform, uniform::SampleUniform};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{Mul, MulAssign},
};

/// A PDF defined by references to an ensemble of particles.
#[derive(Debug)]
pub struct PDFParticles<'a, T, const P: usize>
where
    T: Copy + Scalar,
{
    /// A [`CovMatrix`] that describes an estimate of the underlying particle PDF.
    covmat: CovMatrix<T>,

    /// The particle ensemble that describes the overarching PDF.
    particles: MatrixView<'a, T, Const<P>, Dyn>,

    /// Valid parameter range.
    range: [(T, T); P],

    /// Particle ensemble weights
    weights: &'a Vec<T>,
}

impl<'a, T, const P: usize> PDFParticles<'a, T, P>
where
    T: Copy + RealField + SampleUniform + Scalar + Sum<T> + for<'x> Sum<&'x T>,
{
    /// Access the covariance matrix.
    pub fn covmat(&self) -> &CovMatrix<T> {
        &self.covmat
    }

    /// Create a new [`PDFParticles`] from a particle matrix view.
    pub fn from_particles(
        particles: MatrixView<'a, T, Const<P>, Dyn>,
        range: [(T, T); P],
        weights: &'a Vec<T>,
    ) -> Result<Self, StatsError<T>> {
        let covmat = CovMatrix::from_vectors(&particles.as_view(), Some(weights.as_slice()))?;

        Ok(Self {
            covmat,
            particles,
            range,
            weights,
        })
    }

    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
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
        self.weights
    }
}

impl<T, const P: usize> PDF<T, P> for &PDFParticles<'_, T, P>
where
    T: Float + RealField + SampleUniform + Scalar,
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
            + self.covmat.cholesky_ltm()
                * SVector::<T, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = offset
                + self.covmat.cholesky_ltm()
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
            covmat: self.covmat * rhs,
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
            covmat: self * rhs.covmat,
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
            covmat: self * rhs.covmat,
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
        self.covmat *= rhs;
    }
}

/// Compute new importance weights for `ptpdf` assuming a transition from `ptpdf_from`.
pub fn ptpdf_importance_weighting<T, D, const P: usize>(
    ptpdf_to: &PDFParticles<T, P>,
    ptpdf_from: &PDFParticles<T, P>,
    prior: &D,
) -> Vec<T>
where
    T: Copy + Float + RealField + SampleUniform + Sum<T> + for<'x> Sum<&'x T>,
    D: PDF<T, P>,
{
    let covmat_inv = ptpdf_from.covmat().inverse_matrix();

    let mut weights = ptpdf_to
        .particles
        .par_column_iter()
        .map(|params_new| {
            let value = ptpdf_from
                .particles
                .column_iter()
                .zip(ptpdf_from.weights.iter())
                .map(|(params_old, weight_old)| {
                    let delta = params_new - params_old;

                    Float::exp(
                        Float::ln(*weight_old) - (delta.transpose() * covmat_inv * delta)[(0, 0)],
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

    weights
}
