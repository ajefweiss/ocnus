use crate::{
    stats::{CovMatrix, StatsError, PDF},
    Fp, PMatrix,
};
use nalgebra::{Const, SVector};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

macro_rules! impl_ptpdf_general {
    () => {
        /// Access the covariance matrix.
        pub fn covm(&self) -> &CovMatrix<Const<P>> {
            &self.covm
        }

        /// Returns `true`` if the ensemble contains no elements.
        pub fn is_empty(&self) -> bool {
            self.parts.is_empty()
        }

        /// Access the particle ensemble matrix.
        pub fn particles(&self) -> &PMatrix<Const<P>> {
            &self.parts
        }

        /// Returns the number of particles in the ensemble, also referred to as its 'length'.
        pub fn len(&self) -> usize {
            self.parts.len()
        }

        /// Access the particle weights.
        pub fn weights(&self) -> &Vec<Fp> {
            &self.weights
        }
    };
}

macro_rules! impl_ptpdf_sampler {
    ($self: expr, $rng: expr) => {{
        let normal = Normal::new(0.0, 1.0).unwrap();
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx = $rng.sample(uniform);

                // Abuse try_fold to return particle index early wrapped within Err().
                match $self
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
                    Ok(_) => $self.weights.len() - 1,
                    Err(idx) => idx,
                }
            };

            $self.parts.column(pdx)
        };

        let mut proposal = offset
            + $self.covm.cholesky_ltm()
                * SVector::<Fp, P>::from_iterator((0..P).map(|_| $rng.sample(normal)));

        // Counter for rejected proposals.
        let mut limit = 0;

        while !$self.validate_sample(&proposal.as_view()) {
            if limit > 500 {
                return Err(StatsError::SamplerLimit(500));
            }

            proposal = offset
                + $self.covm.cholesky_ltm()
                    * SVector::<Fp, P>::from_iterator((0..P).map(|_| $rng.sample(normal)));

            limit += 1;
        }

        Ok(proposal)
    }};
}

/// A probability density function (PDF) defined by an ensemble of particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ParticlePDF<const P: usize> {
    /// Covariance matrix describing the ensemble.
    covm: CovMatrix<Const<P>>,

    /// A [`PMatrix`] object that acts as the particle ensemble array.
    parts: PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Particle weights
    weights: Vec<Fp>,
}

impl<const P: usize> ParticlePDF<P> {
    impl_ptpdf_general!();

    /// Creates a [`ParticleRefPDF`] object from a [`ParticlePDF`] object.
    pub fn as_ptpdf_ref(&self, range: [(Fp, Fp); P]) -> Result<ParticleRefPDF<P>, StatsError> {
        ParticleRefPDF::new(None, &self.parts, range, self.weights.as_ref())
    }

    /// Create a new object.
    pub fn new(
        optional_covm: Option<CovMatrix<Const<P>>>,
        parts: PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: Vec<Fp>,
    ) -> Result<Self, StatsError> {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::<Const<P>>::from_particles(&parts, Some(&weights)),
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
    fn relative_density(&self, _x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        unimplemented!()
    }

    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        impl_ptpdf_sampler!(self, rng)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}

/// A probability density function (PDF) defined by a reference to an ensemble of particles.
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
    weights: &'a Vec<Fp>,
}

impl<'a, const P: usize> ParticleRefPDF<'a, P> {
    impl_ptpdf_general!();

    /// Create a new [`ParticleRefPDF`] object.
    pub fn new(
        optional_covm: Option<CovMatrix<Const<P>>>,
        parts: &'a PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: &'a Vec<Fp>,
    ) -> Result<Self, StatsError> {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::<Const<P>>::from_particles(parts, Some(weights)),
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
    fn relative_density(&self, _x: &nalgebra::SVectorView<Fp, P>) -> Fp {
        unimplemented!()
    }

    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        impl_ptpdf_sampler!(self, rng)
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.range
    }
}
