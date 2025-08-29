use crate::{
    base::Model,
    stats::{DensityRange, ParticleDensity},
};
use nalgebra::{Const, DVector, Dyn, Matrix, OMatrix, RealField, SVector, VecStorage};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{io::Write, iter::Sum};

/// A model ensemble object.
///
/// Internally, this is implemented as a fancy [`ParticleDensity`] with a complementary set of
/// vectors that hold the forward model and coordinate system states for each ensemble member.
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
    /// Coordinate system states for each ensemble member.
    pub cs_states: Vec<M::CSST>,

    /// Forward model states for each ensemble member.
    pub fm_states: Vec<M::FMST>,

    /// Probablity density function defined by an ensemble of particles.
    pub ptpdf: ParticleDensity<T, D>,
}

impl<T, M, const D: usize> ModelEnsbl<T, M, D>
where
    T: Copy + RealField,
    M: Model<T, D> + ?Sized,
{
    /// Create a new [`ModelEnsbl`] from a pre-existing particle matrix.
    pub fn from_particles(
        particles: Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>>,
        opt_range: Option<&SVector<DensityRange<T>, D>>,
        opt_weights: Option<DVector<T>>,
    ) -> Self
    where
        T: Copy + RealField + Sum,
        M::FMST: Clone + Default,
        M::CSST: Clone + Default,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let size = particles.ncols();

        Self {
            ptpdf: ParticleDensity::from_vectors(&particles.as_view(), opt_range, opt_weights)
                .unwrap(),
            fm_states: vec![M::FMST::default(); size],
            cs_states: vec![M::CSST::default(); size],
        }
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble.
    pub fn len(&self) -> usize {
        self.ptpdf.len()
    }

    /// Create a new [`ModelEnsbl`] filled with zeros.
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

    /// Serialize this data structure to a file using the JSON5 format.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}
