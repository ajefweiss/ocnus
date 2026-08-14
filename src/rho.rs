//! The magnetometer module provides functionality for modeling and observing magnetic fields.

use bayesfm::{
    EnsembleModel, EnsembleObservations, EnsembleState, ModelError,
    conf::{ConfCamera, ConfPosition},
    noise::Noise,
    obs::{ObsImg, ObsVec},
};
use nalgebra::{
    Const, DMatrix, RealField, SVector, SVectorView, Scalar, U1, U3, Unit, UnitQuaternion, Vector3,
};
use num_traits::AsPrimitive;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardUniform, uniform::SampleUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{iter::Sum, ops::Sub};
use wcs::WCS;

/// A trait that is shared by all models that describe an electron density structure.
pub trait ElectronDensity<T, OC, const P: usize>: EnsembleModel<T, 3, P>
where
    T: RealField + SampleUniform + Sum,
    OC: ConfPosition<T, 3> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
    Self: Sized,
{
    /// Returns an in situ electron density observation.
    fn observe_electron_density(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsVec<T, 1>, ModelError<T>> {
        let position = conf.position();

        let q = match Self::transform_external_to_internal::<U1, U3, _, _>(
            &position.as_view(),
            params,
            cs_state,
        ) {
            Some(value) => value,
            None => {
                return Err(ModelError::Coordinates(position.as_slice().to_vec()));
            }
        };

        // Here we just pass the value through.
        match self.observe_electron_density_ics(&q.as_view(), params, fm_state, cs_state) {
            Some(b_q) => Ok(ObsVec::<T, 1>::from(b_q)),
            None => Ok(ObsVec::<T, 1>::from([(-T::one()).sqrt()])),
        }
    }

    /// Returns the in situ magnetic field vector in internal coordinates.
    fn observe_electron_density_ics(
        &self,
        ics: &SVectorView<T, 3>,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Option<SVector<T, 1>>;

    /// Perform an ensemble forward simulation for an electron density measurement, in parallel, for the given spacecraft observers
    /// and noise model `NM`.
    fn simulate_electron_density<NM>(
        &self,
        ensbl: &mut EnsembleState<T, Self::CSST, Self::FMST, 3, P>,
        obs_ensbl: &mut EnsembleObservations<OC, ObsVec<T, 1>>,
        opt_noise: &mut Option<&mut NM>,
    ) -> Result<(), ModelError<T>>
    where
        OC: Sync,
        NM: Noise<ObsVec<T, 1>> + Sync,
        Self::CSST: Send,
        Self::FMST: Send,
        Self: Sized + Sync,
    {
        self.simulate_ensbl_par(ensbl, obs_ensbl, &Self::observe_electron_density, opt_noise)
    }
}

/// A trait that is shared by all models that describe an electron density structure and can generate remote white-light images.
pub trait ElectronCamera<T, OC, const P: usize>: ElectronDensity<T, OC, P>
where
    T: RealField + SampleUniform + Sum,
    OC: ConfCamera<T> + Scalar + Sync,
    for<'a> &'a OC: Sub<&'a OC, Output = T>,
{
    /// Returns a white-light image generated from the electron density structure.
    fn observe_remote_white_light(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsImg<T>, ModelError<T>>
    where
        T: AsPrimitive<f64>,
        Self::CSST: Sync,
        Self::FMST: Sync,
        Self: Sync,
        StandardUniform: Distribution<T>,
    {
        let wcs = WCS::new(conf.wcs()).unwrap();
        let position = conf.position();

        let image_res_x = wcs.img_dimensions()[0] as usize;
        let image_res_y = wcs.img_dimensions()[1] as usize;
        let samples = 4 * image_res_x * image_res_y;

        let mut image = DMatrix::<T>::zeros(image_res_x, image_res_y);

        let rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let mut positions_ics = DMatrix::from_iterator(
            3,
            samples,
            rng.sample_iter::<T, StandardUniform>(StandardUniform)
                .take(3 * samples),
        );

        // Sampling corrections, we go from mu = [0, 1] to mu = [0, 1.33].
        positions_ics
            .iter_mut()
            .step_by(3)
            .for_each(|ics| *ics = T::from_f64(1.33).unwrap() * *ics);

        let positions_ecs = Vec::from_par_iter(positions_ics.par_column_iter().map(|ics| {
            Self::transform_internal_to_external::<U1, Const<3>, _, _>(
                &ics.as_view(),
                params,
                cs_state,
            )
            .unwrap()
        }));

        // Weight due to density and sampling probability.
        let weights_ics: Vec<_> =
            Vec::from_par_iter(positions_ics.par_column_iter().map(|sample| {
                self.observe_electron_density_ics(&sample.as_view(), params, fm_state, cs_state)
                    .unwrap()[0]
                    * Self::sqrt_detg::<U1, Const<3>, _, _>(&sample.as_view(), params, cs_state)
                        .unwrap()
            }));

        // Weight due to 3D position (distances).
        let mut weights_ecs = Vec::from_par_iter(
            positions_ecs
                .par_iter()
                .map(|ecs| T::one() / (ecs - position).norm().powi(2) / ecs.norm().powi(2)),
        );

        // Weights due to polarization / scattering angle.
        weights_ecs
            .par_iter_mut()
            .zip(positions_ecs.par_iter())
            .for_each(|(weight, ecs)| {
                // Radial polarization component.
                let er =
                    (ecs.dot(&((ecs - position) / ecs.norm() / (ecs - position).norm()))).powi(2);

                *weight *= T::one() + er;

                // match conf.polarization() {
                //     Some(_pol_angle) => {
                //         // TODO: Re-enable polarization calculation.
                //         // // Plane normal to define radial / tangential.
                //         // let mut plane_normal = (ecs - position).cross(ecs);

                //         // // If cross-product is ill-defined.
                //         // if plane_normal.norm() < T::from_f64(1e-6).unwrap() {
                //         //     plane_normal = Vector3::new(T::zero(), T::zero(), T::one());
                //         // } else {
                //         //     plane_normal = plane_normal / plane_normal.norm();
                //         // }

                //         // let polaxis = Vector3::<T>::new(
                //         //     T::from_f64(pol_angle.x).unwrap(),
                //         //     T::from_f64(pol_angle.y).unwrap(),
                //         //     T::from_f64(pol_angle.z).unwrap(),
                //         // );

                //         // let pa_deproj =
                //         //     (polaxis - position * position.dot(&polaxis)).normalize();

                //         // let pn_deproj = (plane_normal
                //         //     - position * position.dot(&plane_normal))
                //         // .normalize();

                //         // let pol = (pa_deproj.dot(&pn_deproj)).acos();

                //         // pol.sin().powi(2) + er * pol.cos().powi(2)

                //         unimplemented!("polarization images currently not supported")
                //     }
                //     None => T::one() + er,
                // }
            });

        // Project external coordinates onto the celestian sphere, as seen by the observer.
        // Build a quaternion to de-rotate the external coordinates into a system with lon/lat.
        let ux = Vector3::<T>::x_axis();
        let rot_axis =
            if (position.normalize() - ux.into_inner()).norm() < T::from_f64(5e-2).unwrap() {
                Vector3::<T>::z_axis()
            } else {
                Unit::new_normalize(ux.cross(&position))
            };
        let rot_angle = (ux.dot(&-position) / position.norm()).acos();
        let rot_q = UnitQuaternion::from_axis_angle(&rot_axis, rot_angle);

        let positions_xy = Vec::from_par_iter(positions_ecs.par_iter().map(|ecs| {
            let derot = if rot_q.coords.x.is_finite() {
                rot_q.transform_vector(&(ecs - position))
            } else {
                ecs - position
            };

            let lon = (-derot.y / derot.x).atan().as_();
            let lat = (derot.z / derot.x).atan().as_();

            // TODO: Fix panic in `wcs` crate.
            let result = std::panic::catch_unwind(|| wcs.proj_lonlat(&wcs::LonLat::new(lon, lat)));

            match result {
                Ok(Some(xy)) => (xy.x(), xy.y()),
                _ => (f64::NAN, f64::NAN),
            }
        }));

        positions_xy
            .iter()
            .zip(weights_ics.iter())
            .zip(weights_ecs.iter())
            .for_each(|(((x, y), &w1), &w2)| {
                let xr = *x as usize;
                let yr = *y as usize;

                if (xr > 0) && (xr < image_res_x - 1) && (yr > 0) && (yr < image_res_y - 1) {
                    let cw = w1 * w2;

                    image[(xr, yr)] += cw;
                }
            });

        Ok(ObsImg(image))
    }
}
