use crate::{
    stats::{CovMatrix, StatsError, PDF},
    Fp, PMatrix,
};
use nalgebra::{
    allocator::{Allocator, Reallocator},
    Const, DefaultAllocator, Dim, DimName, Dyn, OMatrix, SVector, StorageMut, VecStorage,
};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

/// A probability density function (PDF) defined by an ensemble particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticlePDF<const P: usize> {
    /// Covariance matrix describing the ensemble.
    covm: CovMatrix<Const<P>>,

    /// A [`PMatrix`] object that acts as the particle ensemble array.
    parts: PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Optional particle weights
    weights: Option<Vec<Fp>>,
}

impl<const P: usize> ParticlePDF<P> {
    /// Creates a [`ParticleRefPDF`] object from a [`ParticlePDF`] object.
    pub fn as_ptpdf_mut(&mut self, range: [(Fp, Fp); P]) -> Result<ParticleMutPDF<P>, StatsError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        ParticleMutPDF::new(None, &mut self.parts, range, self.weights.as_mut())
    }

    /// Creates a [`ParticleRefPDF`] object from a [`ParticlePDF`] object.
    pub fn as_ptpdf_ref(&self, range: [(Fp, Fp); P]) -> Result<ParticleRefPDF<P>, StatsError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        ParticleRefPDF::new(None, &self.parts, range, self.weights.as_ref())
    }

    /// Access the inverse of the covariance matrix.
    pub fn covm_inverse(&self) -> &OMatrix<Fp, Const<P>, Const<P>> {
        self.covm.inverse()
    }

    /// Create a new [`ParticlePDF`] object.
    pub fn new(
        optional_covm: Option<CovMatrix<Const<P>>>,
        parts: PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: Option<Vec<Fp>>,
    ) -> Result<Self, StatsError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::<Const<P>>::from_particles(&parts, weights.as_ref()),
        }?;

        Ok(Self {
            covm,
            parts,
            range,
            weights,
        })
    }
}

impl<const P: usize> PDF<P> for &ParticlePDF<P> {
    fn relative_likelihood(&self, x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        unimplemented!()
    }

    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        // We circumvent the need to copy the particle sampling code.
        // The compiler "should" optimize away the clone/copy operations.
        let ptpdf_ref = ParticleRefPDF {
            covm: self.covm.clone(),
            parts: &self.parts,
            range: self.range,
            weights: self.weights.as_ref(),
        };

        (&ptpdf_ref).sample(rng)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}

/// A probability density function (PDF) defined by a reference to an ensemble particles.
#[derive(Clone, Debug, Serialize)]
pub struct ParticleRefPDF<'a, const P: usize> {
    /// Covariance matrix describing the ensemble.
    covm: CovMatrix<Const<P>>,

    /// A reference to a [`PMatrix`] object that acts as the particle ensemble array.
    parts: &'a PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Optional particle weights
    weights: Option<&'a Vec<Fp>>,
}

impl<'a, const P: usize> ParticleRefPDF<'a, P> {
    /// Access the inverse of the covariance matrix.
    pub fn covm_inverse(&self) -> &OMatrix<Fp, Const<P>, Const<P>> {
        self.covm.inverse()
    }

    /// Create a new [`ParticleRefPDF`] object.
    pub fn new(
        optional_covm: Option<CovMatrix<Const<P>>>,
        parts: &'a PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: Option<&'a Vec<Fp>>,
    ) -> Result<Self, StatsError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::<Const<P>>::from_particles(parts, weights),
        }?;

        Ok(Self {
            covm,
            parts,
            range,
            weights,
        })
    }
}

impl<'a, const P: usize> PDF<P> for &'a ParticleRefPDF<'a, P> {
    fn relative_likelihood(&self, x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        unimplemented!()
    }

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

        while !self.validate_sample(&proposal.as_view()) {
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

/// A probability density function (PDF) defined by a mutable reference to an ensemble particles.
#[derive(Debug, Serialize)]
pub struct ParticleMutPDF<'a, const P: usize> {
    /// Covariance matrix describing the ensemble.
    covm: CovMatrix<Const<P>>,

    /// A mutable reference to a [`PMatrix`] object that acts as the particle ensemble array.
    parts: &'a mut PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Optional particle weights
    weights: Option<&'a mut Vec<Fp>>,
}

impl<'a, const P: usize> ParticleMutPDF<'a, P> {
    /// Access the inverse of the covariance matrix.
    pub fn covm_inverse(&self) -> &OMatrix<Fp, Const<P>, Const<P>> {
        self.covm.inverse()
    }

    /// Create a new [`ParticleMutPDF`] object.
    pub fn new(
        optional_covm: Option<CovMatrix<Const<P>>>,
        parts: &'a mut PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: Option<&'a mut Vec<Fp>>,
    ) -> Result<Self, StatsError>
    where
        Const<P>: Dim + DimName,
        DefaultAllocator: Allocator<Const<P>>
            + Allocator<Const<P>, Const<P>>
            + Allocator<Const<P>, Const<P>, Buffer<Fp> = VecStorage<Fp, Const<P>, Const<P>>>
            + Reallocator<Fp, Const<P>, Const<P>, Const<P>, Dyn>,
        <DefaultAllocator as Allocator<Const<P>, Const<P>>>::Buffer<Fp>:
            for<'b> Deserialize<'b> + Serialize,
        VecStorage<Fp, Const<P>, Const<P>>: StorageMut<Fp, Const<P>, Const<P>>,
    {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::<Const<P>>::from_particles(parts, weights.as_deref()),
        }?;

        Ok(Self {
            covm,
            parts,
            range,
            weights,
        })
    }
}

impl<'a, const P: usize> PDF<P> for &'a ParticleMutPDF<'a, P> {
    fn relative_likelihood(&self, x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        unimplemented!()
    }

    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        // We circumvent the need to copy the particle sampling code.
        // The compiler "should" optimize away the clone/copy operations.
        let ptpdf_ref = ParticleRefPDF {
            covm: self.covm.clone(),
            parts: self.parts,
            range: self.range,
            weights: self.weights.as_deref(),
        };

        (&ptpdf_ref).sample(rng)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}
