//! Method traits.

mod abc;
mod fisher;
mod gpf;
mod sir;

pub use abc::{ABCParticleFilter, ABCSettings};
pub use fisher::FisherInformation;
pub use gpf::{
    ParticleFilter, ParticleFilterError, ParticleFilterResults, ParticleFilterSettings,
    ParticleFilterSettingsBuilder,
};
pub use sir::SIRParticleFilter;

use crate::{
    OcnusModel,
    obser::{ObserVec, OcnusObser},
};
use nalgebra::{RealField, Scalar};
use num_traits::{AsPrimitive, Zero};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::{iter::Sum, ops::AddAssign};

// Blanket implementation of the [`ABCParticleFilter`] trait for all models.
impl<T, O, const P: usize, FMST, CSST, M> ABCParticleFilter<T, O, P, FMST, CSST> for M
where
    M: ParticleFilter<T, O, P, FMST, CSST>,
    T: AsPrimitive<usize> + Copy + RealField + SampleUniform + Sum + for<'x> Sum<&'x T>,
    O: AddAssign + OcnusObser + Scalar + Zero,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
}

// Blanket implementation of the [`FisherInformation`] trait for all models with [`ObserVec`] observables.
impl<T, const P: usize, const N: usize, FMST, CSST, M> FisherInformation<T, P, N, FMST, CSST> for M
where
    M: OcnusModel<T, ObserVec<T, N>, P, FMST, CSST>,
    T: Copy + RealField + Sum,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
}

// Blanket implementation of the [`ParticleFilter`] trait for all models.
impl<T, O, const P: usize, FMST, CSST, M> ParticleFilter<T, O, P, FMST, CSST> for M
where
    M: OcnusModel<T, O, P, FMST, CSST>,
    T: AsPrimitive<usize> + Copy + RealField,
    O: AddAssign + OcnusObser + Scalar + Zero,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
}

// Blanket implementation of the [`SIRParticleFilter`] trait for all models with [`ObserVec`] observables.
impl<T, const P: usize, const N: usize, FMST, CSST, M> SIRParticleFilter<T, P, N, FMST, CSST> for M
where
    M: ParticleFilter<T, ObserVec<T, N>, P, FMST, CSST>,
    T: AsPrimitive<usize> + Copy + RealField + SampleUniform + Sum + for<'x> Sum<&'x T>,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    Self: Sync,
{
}
