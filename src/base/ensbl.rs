use crate::{
    base::Model,
    stats::{DensityRange, ParticleDensity},
};
use nalgebra::{Const, Dyn, Matrix, OMatrix, RealField, SVector, VecStorage};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum};

/// A model parameter ensemble.
///
/// Internally, this is just a fancy [`ParticleDensity`]
/// with complementary forward model and coordinate system states.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize,
    M::FMST: Serialize,
    M::CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    M::FMST: Deserialize<'de>,
    M::CSST: Deserialize<'de>"))]
pub struct ModelEnsbl<T, M, const D: usize>
where
    T: Copy + RealField,
    M: Model<T, D> + ?Sized,
{
    /// The forward coordinate system states for each ensemble member.
    pub cs_states: Vec<M::CSST>,

    /// The forward model states for each ensemble member.
    pub fm_states: Vec<M::FMST>,

    /// Probablity density function defined by an ensemble of particles.
    pub ptpdf: ParticleDensity<T, D>,
}

impl<T, M, const D: usize> ModelEnsbl<T, M, D>
where
    T: Copy + RealField,
    M: Model<T, D> + ?Sized,
{
    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a new [`ModelEnsbl`] from a pre-existing particle matrix.
    pub fn from_particles(particles: Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>>) -> Self
    where
        T: Copy + RealField + Sum,
        M::FMST: Clone + Default,
        M::CSST: Clone + Default,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let size = particles.ncols();

        Self {
            ptpdf: ParticleDensity::from_vectors(&particles.as_view(), None, None).unwrap(),
            fm_states: vec![M::FMST::default(); size],
            cs_states: vec![M::CSST::default(); size],
        }
    }

    /// Returns the number of members in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.ptpdf.len()
    }

    /// Create a new [`ModelEnsbl`] filled with zeros and un-bounded range.
    pub fn new(size: usize, opt_range: Option<&SVector<DensityRange<T>, D>>) -> Self
    where
        T: Copy + RealField + Sum,
        M::FMST: Clone + Default,
        M::CSST: Clone + Default,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        Self {
            ptpdf: ParticleDensity::from_vectors(
                &OMatrix::<T, Const<D>, Dyn>::zeros(size).as_view(),
                opt_range,
                None,
            )
            .unwrap(),
            fm_states: vec![M::FMST::default(); size],
            cs_states: vec![M::CSST::default(); size],
        }
    }

    /// Store the model ensemble in a JSON5 file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}
