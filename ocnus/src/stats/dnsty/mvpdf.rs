use crate::{
    stats::dnsty::{DensityError, ProbabilityDensityFunctionSampling},
    Fp, PVector, ParamVectorsViewMut,
};
use nalgebra::{
    allocator::{Allocator, Reallocator},
    ArrayStorage, Const, DVector, DefaultAllocator, Dim, DimAdd, Dyn, SVector, VecStorage, U1,
};
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

use super::cvmat::CovMatrix;

/// A multivariate normal probability density function (PDF).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MultivariateProbabilityDensityFunction<D>
where
    Const<D>: Dim + DimAdd<U1>,
    DefaultAllocator: Allocator<Const<D>>
        + Allocator<U1, Const<D>>
        + Allocator<Const<D>, Const<D>>
        + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
    <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
        for<'a> Deserialize<'a> + Serialize,
{
    /// Covariance matrix that describes the multivariate normal PDF.
    // pub covariance: CovMatrix<D>,

    /// The mean parameter of the PDF.
    pub mean: PVector<D>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    pub range: [(Fp, Fp); D],
}

// impl<D> ProbabilityDensityFunctionSampling<D>
//     for MultivariateProbabilityDensityFunction<D>
// where
//     Const<D>: Dim + DimAdd<U1>,
//     DefaultAllocator: Allocator<Const<D>>
//         + Allocator<U1, Const<D>>
//         + Allocator<Const<D>, Const<D>>
//         + Reallocator<Fp, Const<D>, Const<D>, Const<D>, Dyn>,
//     <DefaultAllocator as Allocator<Const<D>, Const<D>>>::Buffer<Fp>:
//         for<'a> Deserialize<'a> + Serialize,
// {
//     fn sample_fill<RStride: Dim, CStride: Dim>(
//         &self,
//         view: &mut ParamVectorsViewMut<D, Dyn, RStride, CStride>,
//         rng: &mut impl rand::Rng,
//     ) -> Result<(), DensityError> where {
//         let normal = Normal::new(0.0, 1.0).unwrap();

//         let x = SVector::from_vec((0..D).map(|_| rng.sample(normal)).collect::<Vec<Fp>>());

//         let mut proposal = self.mean;

//         // Counter for rejected proposals.
//         let mut limit = 0;

//         while !self.validate_candidate(&proposal.as_view()) {
//             if limit > 100 {
//                 return Err(DensityError::ReachedSamplerLimit(100));
//             }

//             proposal =
//                 self.mean + SVector::<Fp, D>::from_data(VecStorage([[rng.sample(normal); D]]));

//             limit += 1;
//         }

//         Ok(proposal)
//     }

//     fn valid_range(&self) -> [(Fp, Fp); D] {
//         self.range
//     }
// }
