use crate::{
    alias::PMatrixViewMut,
    stats::{CovMatrix, ProbabilityDensityFunctionSampling, StatsError},
    Fp,
};
use nalgebra::{Const, Dim, Dyn, SVector, U1};
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A multivariate normal probability density function (PDF).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariatePDF<const P: usize> {
    // CovMatrix matrix that describes the multivariate normal ProbabilityDensityFunction.
    pub covm: CovMatrix<Const<P>>,

    /// The mean parameter of the ProbabilityDensityFunction.
    pub mean: SVector<Fp, P>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    pub range: [(Fp, Fp); P],
}

impl<const P: usize> ProbabilityDensityFunctionSampling<P> for &MultivariatePDF<P> {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        seed: u64,
    ) -> Result<(), StatsError> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        pmatrix
            .par_column_iter_mut()
            .chunks(128)
            .enumerate()
            .try_for_each(|(cdx, mut chunk)| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed + 13 * cdx as u64);

                chunk.iter_mut().try_for_each(|col| {
                    let mut proposal = self.mean
                        + self.covm.cholesky
                            * SVector::<Fp, P>::from_iterator((0..P).map(|_| rng.sample(normal)));

                    // Counter for rejected proposals.
                    let mut limit = 0;

                    while !self.validate_pvector::<U1, Const<P>>(&proposal.as_view()) {
                        if limit > 100 {
                            return Err(StatsError::ReachedSamplerLimit(100));
                        }

                        proposal = self.mean
                            + self.covm.cholesky
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
