use crate::stats::{CovMatrix, PDF, StatsError};
use log::warn;
use nalgebra::{Const, Dyn, MatrixView, MatrixViewMut, SVector, SVectorView};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;

/// A PDF defined by a view of a matrix of ensemble particles.
#[derive(Debug)]
pub struct PDFParticles<'a, const P: usize> {
    /// A [`CovMatrix`] that describes an estimate of the underlying particle PDF.
    covmat: CovMatrix,

    /// The particle ensemble that describes the overarching PDF.
    particles: MatrixViewMut<'a, f32, Const<P>, Dyn>,

    /// Valid parameter range.
    range: [(f32, f32); P],

    /// Particle ensemble weights
    weights: Vec<f32>,
}

impl<'a, const P: usize> PDFParticles<'a, P> {
    /// Access the covariance matrix.
    pub fn covmat(&self) -> &CovMatrix {
        &self.covmat
    }

    /// Create a new [`PDFParticles`] from a particle matrix view.
    pub fn from_particles(
        particles: MatrixViewMut<'a, f32, Const<P>, Dyn>,
        range: [(f32, f32); P],
        opt_weights: Option<Vec<f32>>,
    ) -> Result<Self, StatsError> {
        let (covmat, weights) = match opt_weights {
            Some(weights) => (
                CovMatrix::from_vectors(&particles.as_view(), Some(weights.as_slice()))?,
                weights,
            ),
            None => (
                CovMatrix::from_vectors(&particles.as_view(), None)?,
                vec![1.0 / particles.nrows() as f32; particles.nrows()],
            ),
        };

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
    pub fn particles_ref(&self) -> MatrixView<f32, Const<P>, Dyn> {
        self.particles.as_view()
    }

    /// Mutably access the ensemble particle matrix.
    pub fn particles_mut(&mut self) -> &mut MatrixViewMut<'a, f32, Const<P>, Dyn> {
        &mut self.particles
    }
}

impl<const P: usize> PDF<P> for PDFParticles<'_, P> {
    fn relative_density(&self, _x: &SVectorView<f32, P>) -> f32 {
        unimplemented!()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, StatsError> {
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
                * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut attempts = 0;

        while !self.validate_sample(&proposal.as_view()) {
            proposal = offset
                + self.covmat.cholesky_ltm()
                    * SVector::<f32, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            attempts += 1;

            if (attempts > 250) && (attempts % 250 == 0) {
                warn!(
                    "PDFParticles::draw_sample has failed to draw a valid sample after {} tries",
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
pub fn ptpdf_importance_weighting<T: PDF<P>, const P: usize>(
    ptpdf: &mut PDFParticles<P>,
    ptpdf_from: &PDFParticles<P>,
    prior: &T,
) {
    let covmat_inv = ptpdf_from.covmat().inverse_matrix();

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

                    (weight_old.ln() - (delta.transpose() * covmat_inv * delta)[(0, 0)]).exp()
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
