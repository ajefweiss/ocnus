use crate::{
    base::{OcnusModel, OcnusModelError, ScObs, ScObsSeries},
    methods::fisher_information_matrix,
    obser::ObserVec,
};
use nalgebra::{Const, Dim, OVector, RealField, SMatrix, Vector4, VectorView};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

/// A trait that is shared by all models that can measure in situ magnetic fields.
pub trait MeasureInSituMagneticFields<T, const D: usize, FMST, CSST>
where
    T: Copy + RealField + Sum,
{
    /// Compute the fisher information matrix (FIM) using magnetic field vector observations.
    fn fisher_mag<M, CF, RStride: Dim, CStride: Dim>(
        model: &M,
        series: &ScObsSeries<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        acr_func: &CF,
    ) -> Result<SMatrix<T, D, D>, OcnusModelError<T>>
    where
        M: OcnusModel<T, D, FMST, CSST>,
        T: SampleUniform
            + for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        CF: Fn(T, T) -> T + Sync,
        FMST: Clone + Default + Send,
        CSST: Clone + Default + Send,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        fisher_information_matrix(model, series, params, &Self::observe_mag3, acr_func)
    }

    /// Returns an in situ magnetic field vector observation.
    fn observe_mag3(
        scobs: &ScObs<T>,
        params: &OVector<T, Const<D>>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 3>, OcnusModelError<T>>;

    /// Returns an in situ magnetic field vector observation with magnitude.
    fn observe_mag4(
        scobs: &ScObs<T>,
        params: &OVector<T, Const<D>>,
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
}
