use crate::{
    fevms::FEVMEnsbl,
    stats::{CovMatrix, StatsError, PDF},
    Fp, OcnusState, PMatrix,
};
use nalgebra::{Const, SVector};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

macro_rules! impl_ptpdf {
    () => {
        /// Access the covariance matrix.
        pub fn covm(&self) -> &CovMatrix {
            &self.covm
        }

        /// Returns `true`` if the ensemble contains no elements.
        pub fn is_empty(&self) -> bool {
            self.parts.is_empty()
        }

        /// Access the particle ensemble matrix.
        pub fn particles_ref(&self) -> &PMatrix<Const<P>> {
            &self.parts
        }

        /// Returns the number of particles in the ensemble, also referred to as its 'length'.
        pub fn len(&self) -> usize {
            self.parts.len()
        }

        /// Access the particle weights.
        pub fn weights_ref(&self) -> &Vec<Fp> {
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
            if limit > 5000 {
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
    covm: CovMatrix,

    /// A [`PMatrix`] object that acts as the particle ensemble array.
    parts: PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Particle weights
    weights: Vec<Fp>,
}

impl<const P: usize> ParticlePDF<P> {
    impl_ptpdf!();

    /// Creates a [`ParticleRefPDF`] object from a [`ParticlePDF`] object.
    pub fn as_ptpdf_ref(&self) -> Result<ParticleRefPDF<P>, StatsError> {
        ParticleRefPDF::new(None, &self.parts, self.range, self.weights.as_ref())
    }

    /// Update weights assuming a transition from `other` and a prior `prior`.
    pub fn compute_importance_weights(&mut self, other: &ParticleRefPDF<P>, prior: &impl PDF<P>) {
        let covm_inv = other.covm().inverse();

        let mut weights = self
            .parts
            .par_column_iter()
            .map(|params_new| {
                let value = other
                    .particles_ref()
                    .par_column_iter()
                    .zip(other.weights_ref())
                    .map(|(params_old, weight_old)| {
                        let delta = params_new - params_old;

                        (weight_old.ln() - (delta.transpose() * covm_inv * delta)[(0, 0)]).exp()
                    })
                    .sum::<Fp>();

                1.0 / value
            })
            .collect::<Vec<Fp>>();

        let total = weights.iter().sum::<Fp>();

        weights
            .par_iter_mut()
            .zip(self.parts.par_column_iter())
            .for_each(|(weight, params)| *weight *= prior.relative_density(&params) / total);

        self.weights = weights
    }

    /// Convert self into a [`FEVMEnsbl`] object.
    pub fn into_fevme<S: OcnusState>(self) -> FEVMEnsbl<S, P> {
        FEVMEnsbl {
            ensbl: self.parts,
            states: vec![S::default(); self.weights.len()],
            weights: self.weights,
        }
    }

    /// Create a new [`ParticleRefPDF`] object and multiply the covariance matrix by `factor`.
    pub fn mul_covm(&self, factor: Fp) -> Result<ParticleRefPDF<P>, StatsError> {
        let new = ParticleRefPDF::new(None, &self.parts, self.range, self.weights.as_ref())?;

        Ok(new.mul_covm(factor))
    }

    /// Create a new object.
    pub fn new(
        optional_covm: Option<CovMatrix>,
        parts: PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: Vec<Fp>,
    ) -> Result<Self, StatsError> {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::from_particles(&parts, Some(&weights)),
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
    covm: CovMatrix,

    /// A reference to a [`PMatrix`] object that acts as the particle ensemble array.
    parts: &'a PMatrix<Const<P>>,

    /// Valid parameter range.
    #[serde(with = "serde_arrays")]
    range: [(Fp, Fp); P],

    /// Optional particle weights
    weights: &'a Vec<Fp>,
}

impl<'a, const P: usize> ParticleRefPDF<'a, P> {
    impl_ptpdf!();

    /// Create a new [`ParticleRefPDF`] object and multiply the covariance matrix by `factor`.
    pub fn mul_covm(&self, factor: Fp) -> Self {
        Self {
            covm: self.covm.clone() * factor,
            parts: self.parts,
            range: self.range,
            weights: self.weights,
        }
    }

    /// Create a new [`ParticleRefPDF`] object.
    pub fn new(
        optional_covm: Option<CovMatrix>,
        parts: &'a PMatrix<Const<P>>,
        range: [(Fp, Fp); P],
        weights: &'a Vec<Fp>,
    ) -> Result<Self, StatsError> {
        let covm = match optional_covm {
            Some(value) => Ok(value),
            None => CovMatrix::from_particles(parts, Some(weights)),
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
