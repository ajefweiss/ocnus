use crate::stats::{CovMatrix, PDF, StatsError};
use core::panic;
use log::warn;
use nalgebra::{Const, Dyn, MatrixView, SVector, SVectorView};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;
use std::ops::{Mul, MulAssign};

/// A PDF defined by references to an ensemble of particles.
#[derive(Debug)]
pub struct PDFParticles<'a, const P: usize> {
    /// A [`CovMatrix`] that describes an estimate of the underlying particle PDF.
    covmat: CovMatrix,

    /// The particle ensemble that describes the overarching PDF.
    particles: MatrixView<'a, f64, Const<P>, Dyn>,

    /// Valid parameter range.
    range: [(f64, f64); P],

    /// Particle ensemble weights
    weights: &'a Vec<f64>,
}

impl<'a, const P: usize> PDFParticles<'a, P> {
    /// Access the covariance matrix.
    pub fn covmat(&self) -> &CovMatrix {
        &self.covmat
    }

    /// Create a new [`PDFParticles`] from a particle matrix view.
    pub fn from_particles(
        particles: MatrixView<'a, f64, Const<P>, Dyn>,
        range: [(f64, f64); P],
        weights: &'a Vec<f64>,
    ) -> Result<Self, StatsError> {
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
    pub fn particles_ref(&self) -> MatrixView<f64, Const<P>, Dyn> {
        self.particles.as_view()
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &Vec<f64> {
        self.weights
    }
}

impl<const P: usize> PDF<P> for &PDFParticles<'_, P> {
    fn relative_density(&self, _x: &SVectorView<f64, P>) -> f64 {
        unimplemented!()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f64, P>, StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx = rng.sample(uniform);

                // Here we abuse try_fold to return particle index early wrapped within Err().
                match self
                    .weights
                    .iter()
                    .enumerate()
                    .try_fold(0.0, |acc, (idx, weight)| {
                        let next_weight = acc + weight;
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
                * SVector::<f64, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = offset
                + self.covmat.cholesky_ltm()
                    * SVector::<f64, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 250) && (attempts % 250 == 0) {
                warn!(
                    "PDFParticles::draw_sample has failed to draw a valid sample after {} tries",
                    attempts,
                );

                panic!()
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(f64, f64); P] {
        self.range
    }
}

impl<'a, const P: usize> Mul<f64> for PDFParticles<'a, P> {
    type Output = PDFParticles<'a, P>;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            covmat: self.covmat * rhs,
            particles: self.particles,
            range: self.range,
            weights: self.weights,
        }
    }
}

impl<'a, const P: usize> Mul<PDFParticles<'a, P>> for f64 {
    type Output = PDFParticles<'a, P>;

    fn mul(self, rhs: PDFParticles<'a, P>) -> Self::Output {
        Self::Output {
            covmat: self * rhs.covmat,
            particles: rhs.particles,
            range: rhs.range,
            weights: rhs.weights,
        }
    }
}

impl<const P: usize> MulAssign<f64> for PDFParticles<'_, P> {
    fn mul_assign(&mut self, rhs: f64) {
        self.covmat *= rhs;
    }
}

/// Compute new importance weights for `ptpdf` assuming a transition from `ptpdf_from`.
pub fn ptpdf_importance_weighting<T: PDF<P>, const P: usize>(
    ptpdf_to: &PDFParticles<P>,
    ptpdf_from: &PDFParticles<P>,
    prior: &T,
) -> Vec<f64> {
    let covmat_inv = ptpdf_from.covmat().inverse_matrix();

    let mut weights = ptpdf_to
        .particles
        .par_column_iter()
        .map(|params_new| {
            let value = &ptpdf_from
                .particles
                .par_column_iter()
                .zip(ptpdf_from.weights.par_iter())
                .map(|(params_old, weight_old)| {
                    let delta = params_new - params_old;

                    (weight_old.ln() - (delta.transpose() * covmat_inv * delta)[(0, 0)]).exp()
                })
                .sum::<f64>();

            1.0 / value
        })
        .collect::<Vec<f64>>();

    weights
        .par_iter_mut()
        .zip(ptpdf_to.particles.par_column_iter())
        .for_each(|(weight, params)| *weight *= prior.relative_density(&params));

    let weights_total = weights.iter().sum::<f64>();

    weights
        .par_iter_mut()
        .for_each(|weight| *weight /= weights_total);

    weights
}
