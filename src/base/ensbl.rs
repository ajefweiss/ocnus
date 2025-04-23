use crate::stats::{DensityRange, ParticleDensity};
use nalgebra::{Const, Dyn, OMatrix, RealField, SVector};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    iter::Sum,
    ops::{Mul, Sub},
};

/// A model ensemble.
///
/// This is just a fancy [`ParticleDensity`].
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "T: Serialize, FMST: Serialize, CSST: Serialize"))]
#[serde(bound(
    deserialize = "T: Deserialize<'de>, FMST: Deserialize<'de>, CSST: Deserialize<'de>"
))]
pub struct OcnusEnsbl<T, const D: usize, FMST, CSST>
where
    T: Copy + RealField,
{
    /// Ensemble model parameter view.
    pub ptpdf: ParticleDensity<T, D>,

    /// Forward model states.
    pub fm_states: Vec<FMST>,

    /// Coordinate system states.
    pub cs_states: Vec<CSST>,
}

impl<T, const D: usize, FMST, CSST> OcnusEnsbl<T, D, FMST, CSST>
where
    T: Copy + RealField + SampleUniform,
{
    /// Return a mutable reference to the underlying [`ParticleDensity`].
    pub fn as_density_mut(&mut self) -> &mut ParticleDensity<T, D> {
        &mut self.ptpdf
    }

    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.ptpdf.len()
    }

    /// Create a new [`OcnusEnsbl`] filled with zeros.
    pub fn new(size: usize, range: SVector<DensityRange<T>, D>) -> Self
    where
        T: Copy
            + for<'x> Mul<&'x T, Output = T>
            + RealField
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        FMST: Clone + Default,
        CSST: Clone + Default,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        Self {
            ptpdf: ParticleDensity::from_vectors(
                &OMatrix::<T, Const<D>, Dyn>::zeros(size).as_view(),
                range,
                None,
            )
            .unwrap(),
            fm_states: vec![FMST::default(); size],
            cs_states: vec![CSST::default(); size],
        }
    }

    /// Serialize ensemble to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}
