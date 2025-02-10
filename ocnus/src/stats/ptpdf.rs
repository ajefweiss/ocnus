use crate::covmat::CovMatrix;
use log::warn;
use nalgebra::{Const, Dyn, Matrix, SVector, VecStorage};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{OcnusStatsError, PDF};

/// A probability density function (PDF) defined by a reference to an ensemble of particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticlePDF<const P: usize> {
    /// A [`CovMatrix`] that describes an estimate of the underlying particle PDF.
    covm: CovMatrix,

    /// The particle ensemble that describes the overarching PDF.
    particles: Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(f32, f32); P],

    /// Particle ensemble weights
    weights: Vec<f32>,
}

impl<const P: usize> ParticlePDF<P> {
    /// Access the covariance matrix.
    pub fn covm(&self) -> &CovMatrix {
        &self.covm
    }

    /// Returns `true`` if the ensemble contains no elements.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }

    /// Returns the number of particles in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Create a new [`ParticlePDF`].
    pub fn new(
        particles: Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>,
        range: [(f32, f32); P],
        weights: Vec<f32>,
    ) -> Self {
        let covm = CovMatrix::from_particles(&particles, Some(&weights)).unwrap();

        Self {
            covm,
            particles,
            range,
            weights,
        }
    }

    /// Access the ensemble particle matrix.
    pub fn particles_ref(&self) -> &Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>> {
        &self.particles
    }

    /// Mutably access the ensemble particle matrix.
    pub fn particles_mut(
        &mut self,
    ) -> &mut Matrix<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>> {
        &mut self.particles
    }
}

impl<const P: usize> PDF<P> for &ParticlePDF<P> {
    fn relative_density(&self, _x: &nalgebra::SVectorView<f32, P>) -> f32 {
        unimplemented!()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, OcnusStatsError> {
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
            + self.covm.cholesky_ltm()
                * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = offset
                + self.covm.cholesky_ltm()
                    * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 250) && (attempts % 250 == 0) {
                warn!(
                    "ParticlePDF::draw_sample has failed to draw a valid sample after {} tries",
                    attempts
                );
            }
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(f32, f32); P] {
        self.range
    }
}

/// Compute new importance weights for `ptpdf` assuming a transition from `ptpdf_from`.
pub fn compute_importance_weights<const P: usize>(
    ptpdf: &mut ParticlePDF<P>,
    ptpdf_from: &ParticlePDF<P>,
    prior: &impl PDF<P>,
) {
    let covm_inv = ptpdf_from.covm().inverse();

    let mut weights = ptpdf
        .particles
        .par_column_iter()
        .map(|params_new| {
            let value = &ptpdf_from
                .particles
                .par_column_iter()
                .zip(&ptpdf_from.weights)
                .map(|(params_old, weight_old)| {
                    let delta = params_new - params_old;

                    (weight_old.ln() - (delta.transpose() * covm_inv * delta)[(0, 0)]).exp()
                })
                .sum::<f32>();

            1.0 / value
        })
        .collect::<Vec<f32>>();

    weights
        .par_iter_mut()
        .zip(ptpdf.particles.par_column_iter())
        .for_each(|(weight, params)| *weight *= prior.relative_density(&params));

    let weights_total = weights.iter().sum::<f32>();

    weights
        .par_iter_mut()
        .for_each(|weight| *weight /= weights_total);

    ptpdf.weights = weights
}
