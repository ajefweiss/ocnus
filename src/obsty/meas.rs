use crate::{
    base::{Model, ModelError, ScConf, ScObs},
    math::CovMatrix,
    methods::fisher_information_matrix,
    obsty::ObserVec,
};
use nalgebra::{Const, Dim, Dyn, RealField, SMatrix, SVector, Vector4, VectorView};
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::iter::Sum;

/// A trait that is shared by all models that can measure in situ magnetic fields.
pub trait MeasInSituMag<T, const D: usize>
where
    T: Copy + RealField + SampleUniform + Sum,
    Self: Model<T, D>,
{
    /// Compute the fisher information matrix (FIM) using magnetic field vector observations.
    fn fisher_mag<RStride: Dim, CStride: Dim>(
        &self,
        scobs: &ScObs<T, ObserVec<T, 3>>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        covm: &CovMatrix<T, Dyn>,
    ) -> Result<SMatrix<T, D, D>, ModelError<T>>
    where
        T: Float + Sum,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
        Self: Send + Sync,
        Self::CSST: Clone + Default + Send,
        Self::FMST: Clone + Default + Send,
    {
        fisher_information_matrix(self, scobs, params, &Self::observe_mag3, covm)
    }

    /// Returns an in situ magnetic field vector observation.
    fn observe_mag3(
        &self,
        scobs: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 3>, ModelError<T>>;

    /// Returns an in situ magnetic field vector observation with magnitude.
    fn observe_mag4(
        &self,
        scobs: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 4>, ModelError<T>> {
        let measurement = Self::observe_mag3(self, scobs, params, fm_state, cs_state)?;

        Ok(ObserVec::<T, 4>::from(Vector4::from([
            measurement.sum_of_squares().sqrt(),
            measurement[0],
            measurement[1],
            measurement[2],
        ])))
    }
}

/// A trait that is shared by all models that can measure the in situ plasma bulk velocity.
pub trait MeasInSituPBV<T, const D: usize, FMST, CSST>
where
    T: Copy + RealField,
    Self: Model<T, D> + Sized,
{
    /// Returns the in situ plasma bulk velocity.
    fn observe_pbv(
        &self,
        scobs: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &FMST,
        cs_state: &CSST,
    ) -> Result<ObserVec<T, 1>, ModelError<T>>;
}
