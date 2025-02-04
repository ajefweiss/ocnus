use crate::{
    alias::{PMatrix, PMatrixViewMut},
    stats::{CovMatrix, ProbabilityDensityFunctionSampling, StatsError},
    Fp,
};
use nalgebra::{Const, Dim, Dyn, SVector, U1};
use rand::{Rng, SeedableRng};
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A probability density function (PDF) defined by an ensemble particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticlePDF<const P: usize> {
    /// Optional covariance matrix describing the ensemble.
    pub covm: Option<CovMatrix<Const<P>>>,

    /// Underlying particle ensemble.
    pub parts: PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    pub range: [(Fp, Fp); P],

    /// Optional particle weights
    pub weights: Option<Vec<Fp>>,
}

impl<const P: usize> ProbabilityDensityFunctionSampling<P> for &ParticlePDF<P> {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        seed: u64,
    ) -> Result<(), StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        // Unwrap covariance object, return an error otherwise.
        let covm = match self.covm.as_ref() {
            Some(value) => value,
            None => return Err(StatsError::MissingCovMat),
        };

        pmatrix
            .par_column_iter_mut()
            .chunks(256)
            .enumerate()
            .try_for_each(|(cdx, mut chunks)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + 17 * cdx as u64);

                chunks.iter_mut().try_for_each(|col| {
                    let offset = {
                        let pdx = match &self.weights {
                            Some(weights) => {
                                // Select particle index by weight.
                                let wdx = rng.sample(uniform);

                                // Abuse try_fold to return particle index early wrapped within Err().
                                match weights.iter().enumerate().try_fold(
                                    0.0,
                                    |acc, (idx, weight)| {
                                        let next_weight = acc + weight;
                                        if wdx < next_weight {
                                            Err(idx)
                                        } else {
                                            Ok(next_weight)
                                        }
                                    },
                                ) {
                                    Ok(_) => self.weights.as_ref().unwrap().len() - 1,
                                    Err(idx) => idx,
                                }
                            }
                            None => rng.sample(Uniform::new(0, self.parts.len()).unwrap()),
                        };

                        self.parts.column(pdx)
                    };

                    let mut proposal = offset
                        + covm.cholesky
                            * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

                    // Counter for rejected proposals.
                    let mut limit = 0;

                    while !self.validate_pvector::<U1, Const<P>>(&proposal.as_view()) {
                        if limit > 500 {
                            return Err(StatsError::ReachedSamplerLimit(500));
                        }

                        proposal = offset
                            + covm.cholesky
                                * SVector::<Fp, P>::from_iterator(
                                    (0..P).map(|_| rng.sample(normal)),
                                );

                        limit += 1;
                    }

                    col.set_column(0, &proposal);

                    Ok(())
                })?;

                Ok(())
            })?;

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}
