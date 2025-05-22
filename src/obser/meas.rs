use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsSeries},
    obser::ObserVec,
    stats::DensityRange,
};
use covmatrix::CovMatrix;
use nalgebra::{
    Const, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn, OMatrix,
    OVector, RealField, U1, Vector4, VectorView,
    allocator::Allocator,
    constraint::{DimEq, ShapeConstraint},
};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::iter::Sum;

/// A trait that is shared by all models that can measure in situ magnetic fields.
pub trait MeasureInSituMagneticFields<T, const N: usize, FMST, CSST>
where
    T: Copy + RealField + Sum,
{
    /// Returns an in situ magnetic field vector observation.
    fn observe_mag3(
        scobs: &ScObs<T>,
        params: &OVector<T, Const<N>>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 3>, OcnusModelError<T>>;

    /// Returns an in situ magnetic field vector observation with magnitude.
    fn observe_mag4(
        scobs: &ScObs<T>,
        params: &OVector<T, Const<N>>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 4>, OcnusModelError<T>> {
        let measurement = Self::observe_mag3(scobs, params, fm_state, cs_state)?;

        Ok(ObserVec::<T, 4>::from(Vector4::from([
            measurement.sum_of_squares().sqrt(),
            measurement[0],
            measurement[1],
            measurement[2],
        ])))
    }

    /// Compute the FIM using in situ field vector observations and a
    /// time and distance dependent auto correlation function `acr_func`.
    fn fisher_mag3<M, D, CF, RStride, CStride>(
        series: &ScObsSeries<T>,
        model: &M,
        params: &VectorView<T, D, RStride, CStride>,
        acr_func: &CF,
    ) where
        T: SampleUniform,
        M: OcnusModel<T, D, FMST, CSST>,
        D: DimName + DimMin<D>,
        FMST: Clone + Default + Send,
        CSST: Clone + Default + Send,
        CF: Fn(T, T) -> T + Sync,
        RStride: Dim,
        CStride: Dim,
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
        <DefaultAllocator as Allocator<D>>::Buffer<T>: Sync,
        <DefaultAllocator as Allocator<D, Dyn>>::Buffer<T>: Send + Sync,
        <DefaultAllocator as Allocator<D, D>>::Buffer<T>: Sync,
        <DefaultAllocator as Allocator<D>>::Buffer<DensityRange<T>>: Sync,
        usize: AsPrimitive<T>,
        ShapeConstraint: DimEq<D, Const<N>>,
    {
        let step_sizes = model.param_step_sizes();
        let result = OMatrix::<T, D, D>::zeros();

        ()
    }
}
