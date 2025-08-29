use crate::{
    base::{Model, ModelError, ScConf, ScObs},
    math::CovMatrix,
    methods::fisher_information_matrix,
    obsty::{ObserImg, ObserVec},
};
use nalgebra::{
    Const, DMatrix, Dim, Dyn, RealField, SMatrix, SVector, Scalar, U1, U3, Vector4, VectorView,
    VectorView3,
};
use num_traits::{AsPrimitive, Float};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::iter::Sum;

/// A trait that is shared by all models that can measure the in situ plasma density.
pub trait InSituPlasmaDensity<T, const D: usize>
where
    T: Copy + RealField,
    Self: Model<T, D> + Sized,
{
    /// Returns the in situ plasma density.
    fn observe_np(
        &self,
        scconf: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 1>, ModelError<T>> {
        let sc_pos = scconf.position();

        let q = match Self::transform_ecs_to_ics(
            &sc_pos.as_view(),
            &params.generic_view((0, 0), (Const::<D>, Const::<1>)),
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::CoordinateTransform(sc_pos.into_owned()));
            }
        };

        Ok(ObserVec::from([self.observe_np_ics(
            &q.as_view(),
            params,
            fm_state,
            cs_state,
        )]))
    }

    /// Returns the in situ plasma density using internal coordinates.
    fn observe_np_ics(
        &self,
        ics: &VectorView3<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> T;
}

/// A trait that is shared by all models that can measure in situ magnetic fields.
pub trait InSituMagnetometer<T, const D: usize>
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
        scconf: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 3>, ModelError<T>>;

    /// Returns an in situ magnetic field vector observation with magnitude.
    fn observe_mag4(
        &self,
        scconf: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 4>, ModelError<T>> {
        let measurement = Self::observe_mag3(self, scconf, params, fm_state, cs_state)?;

        Ok(ObserVec::<T, 4>::from(Vector4::from([
            measurement.sum_of_squares().sqrt(),
            measurement[0],
            measurement[1],
            measurement[2],
        ])))
    }
}

/// A trait that is shared by all models that can measure the in situ plasma bulk velocity.
pub trait InSituPlasmaBulkVelocity<T, const D: usize>
where
    T: Copy + RealField,
    Self: Model<T, D> + Sized,
{
    /// Returns the in situ plasma bulk velocity.
    fn observe_pbv(
        &self,
        scconf: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserVec<T, 1>, ModelError<T>>;
}

/// A trait that is shared by all models from which one can generate synthetic remote white light images.
pub trait RemoteWhiteLight<T, const D: usize>
where
    T: AsPrimitive<usize> + Copy + RealField + Scalar,
    Self: Model<T, D> + InSituPlasmaDensity<T, D> + Sized + Sync,
    Self::FMST: Sync,
    Self::CSST: Sync,
    StandardUniform: Distribution<T>,
{
    /// Returns the remote white-light observation.
    fn observe_rwl(
        &self,
        scconf: &ScConf<T>,
        params: &SVector<T, D>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObserImg<T>, ModelError<T>> {
        let resolution = scconf.resolution();
        let mut image = DMatrix::<T>::zeros(resolution, resolution);

        let rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let mut positions_ics = DMatrix::from_iterator(
            3,
            scconf.samples(),
            rng.sample_iter::<T, StandardUniform>(StandardUniform)
                .take(3 * scconf.samples()),
        );

        // Sampling corrections
        positions_ics
            .iter_mut()
            .step_by(3)
            .for_each(|ics| *ics = ics.sqrt());

        let positions_ecs = Vec::from_par_iter(positions_ics.par_column_iter().map(|ics| {
            Self::transform_ics_to_ecs(
                &ics.as_view(),
                &params.as_view::<Const<D>, U1, U1, Const<D>>(),
                cs_state,
            )
            .unwrap()
        }));

        // Weight due to density and sampling probability.
        let weights_ics = Vec::from_par_iter(positions_ics.par_column_iter().map(|sample| {
            self.observe_np_ics(
                &sample.as_view::<U3, U1, U1, U3>(),
                params,
                fm_state,
                cs_state,
            ) * Self::detg(
                &sample.as_view::<U3, U1, U1, U3>(),
                &params.as_view::<Const<D>, U1, U1, Const<D>>(),
                cs_state,
            )
            .unwrap()
        }));

        // Weight due to 3D position (distance & angles.)
        let weights_ecs = Vec::from_par_iter(
            positions_ecs
                .par_iter()
                .map(|ecs| T::one() / (ecs - scconf.position()).norm().powi(2)),
        );

        let positions_xy =
            Vec::from_par_iter(positions_ecs.par_iter().map(|ecs| scconf.project(ecs)));

        positions_xy
            .iter()
            .zip(weights_ics.iter())
            .zip(weights_ecs.iter())
            .for_each(|(((x, y), &w1), &w2)| {
                let xr: usize = x.as_();
                let yr: usize = y.as_();

                if (xr > 0)
                    && (xr < scconf.resolution() - 1)
                    && (yr > 0)
                    && (yr < scconf.resolution() - 1)
                {
                    let cw = w1 * w2;

                    image[(xr, yr)] += cw;
                }
            });

        Ok(ObserImg(image))
    }
}
