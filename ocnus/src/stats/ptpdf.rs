use crate::{
    stats::{CovMatrix, ProbabilityDensityFunctionSampling, StatsError},
    Fp, PMatrix,
};
use nalgebra::{Const, SVector};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

/// A probability density function (PDF) defined by an ensemble particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticlePDF<const P: usize> {
    /// Optional covariance matrix describing the ensemble.
    covm: CovMatrix<Const<P>>,

    /// Underlying particle ensemble.
    parts: PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Optional particle weights
    weights: Option<Vec<Fp>>,
}

impl<const P: usize> ProbabilityDensityFunctionSampling<P> for &ParticlePDF<P> {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let offset = {
            let pdx = match &self.weights {
                Some(weights) => {
                    // Select particle index by weight.
                    let wdx = rng.sample(uniform);

                    // Abuse try_fold to return particle index early wrapped within Err().
                    match weights
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
                        Ok(_) => self.weights.as_ref().unwrap().len() - 1,
                        Err(idx) => idx,
                    }
                }
                None => rng.sample(Uniform::new(0, self.parts.len()).unwrap()),
            };

            self.parts.column(pdx)
        };

        let mut proposal = offset
            + self.covm.cholesky_ltm()
                * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

        // Counter for rejected proposals.
        let mut limit = 0;

        while !self.validate_sample(&proposal) {
            if limit > 500 {
                return Err(StatsError::SamplerLimit(500));
            }

            proposal = offset
                + self.covm.cholesky_ltm()
                    * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

            limit += 1;
        }

        Ok(proposal)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}
