use crate::stats::{DensityRange, ParticleDensity};
use nalgebra::{
    DefaultAllocator, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn, OMatrix, OVector,
    RealField, U1, allocator::Allocator,
};
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
#[serde(bound(serialize = "
    T: Serialize, 
    OVector<T, D>: Serialize, 
    OVector<DensityRange<T>, D>: Serialize, 
    OMatrix<T, D, D>: Serialize, 
    OMatrix<T, D, Dyn>: Serialize,
    FMST: Serialize,
    CSST: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    OVector<T, D>: Deserialize<'de>, 
    OVector<DensityRange<T>, D>: Deserialize<'de>, 
    OMatrix<T, D, D>: Deserialize<'de> , 
    OMatrix<T, D, Dyn>: Deserialize<'de>,
    FMST: Deserialize<'de>,
    CSST: Deserialize<'de>"))]
pub struct OcnusEnsbl<T, D, FMST, CSST>
where
    T: Copy + RealField,
    D: DimName + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    /// Ensemble model parameter view.
    pub ptpdf: ParticleDensity<T, D>,

    /// Forward model states.
    pub fm_states: Vec<FMST>,

    /// Coordinate system states.
    pub cs_states: Vec<CSST>,
}

impl<T, D, FMST, CSST> OcnusEnsbl<T, D, FMST, CSST>
where
    T: Copy + RealField + SampleUniform,
    D: DimName + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
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
    pub fn new(size: usize, range: OVector<DensityRange<T>, D>) -> Self
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
        usize: AsPrimitive<T>,
    {
        Self {
            ptpdf: ParticleDensity::from_vectors(OMatrix::<T, D, Dyn>::zeros(size), range, None)
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
