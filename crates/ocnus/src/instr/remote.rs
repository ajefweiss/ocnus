use crate::{
    base::{Model, ModelError},
    instr::Plasma,
    obs::{
        conf::{ObsCam, ObsTime},
        data::ObsImg,
    },
};
use nalgebra::{Const, DMatrix, RealField, SVectorView, Scalar, U1, Unit, UnitQuaternion, Vector3};
use num_traits::AsPrimitive;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardUniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use wcs::WCS;

/// A trait that is shared by all models from which one can generate synthetic remote white light images.
pub trait WLCamera<T, OC, const P: usize>
where
    T: Copy + RealField,
    OC: ObsTime<T> + ObsCam<T> + Scalar,
    Self: Model<T, 3, P> + Plasma<T, OC, 3, P>,
{
    /// Returns the remote white-light observation.
    fn observe_rwl(
        &self,
        conf: &OC,
        params: &SVectorView<T, P>,
        fm_state: &Self::FMST,
        cs_state: &Self::CSST,
    ) -> Result<ObsImg<T>, ModelError<T>>
    where
        T: AsPrimitive<f64>,
        OC: Sync,
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

        // // Sampling corrections
        // TODO FIX THIS + REQUIRED WEIGHTING
        // positions_ics
        //     .iter_mut()
        //     .step_by(3)
        //     .for_each(|ics| *ics = (T::from_f64(1.33).unwrap() * *ics).sqrt());
        positions_ics
            .iter_mut()
            .step_by(3)
            .for_each(|ics| *ics = T::from_f64(1.33).unwrap() * *ics);

        let positions_ecs = Vec::from_par_iter(positions_ics.par_column_iter().map(|ics| {
            Self::transform_ics_to_ecs::<U1, Const<P>>(&ics.as_view(), params, cs_state).unwrap()
        }));

        // Weight due to density and sampling probability.
        let weights_ics: Vec<_> =
            Vec::from_par_iter(positions_ics.par_column_iter().map(|sample| {
                self.observe_rho_ics(&sample.as_view(), params, fm_state, cs_state)
                    * Self::sqrtdetg::<U1, Const<P>>(&sample.as_view(), params, cs_state).unwrap()
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

                *weight *= match conf.polarization() {
                    Some(_pol_angle) => {
                        // TODO: Re-enable polarization calculation.
                        // // Plane normal to define radial / tangential.
                        // let mut plane_normal = (ecs - position).cross(ecs);

                        // // If cross-product is ill-defined.
                        // if plane_normal.norm() < T::from_f64(1e-6).unwrap() {
                        //     plane_normal = Vector3::new(T::zero(), T::zero(), T::one());
                        // } else {
                        //     plane_normal = plane_normal / plane_normal.norm();
                        // }

                        // let polaxis = Vector3::<T>::new(
                        //     T::from_f64(pol_angle.x).unwrap(),
                        //     T::from_f64(pol_angle.y).unwrap(),
                        //     T::from_f64(pol_angle.z).unwrap(),
                        // );

                        // let pa_deproj =
                        //     (polaxis - position * position.dot(&polaxis)).normalize();

                        // let pn_deproj = (plane_normal
                        //     - position * position.dot(&plane_normal))
                        // .normalize();

                        // let pol = (pa_deproj.dot(&pn_deproj)).acos();

                        // pol.sin().powi(2) + er * pol.cos().powi(2)

                        unimplemented!("polarization images currently not supported")
                    }
                    None => T::one() + er,
                }
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

            let lon = (-derot.y / derot.x).atan();
            let lat = (derot.z / derot.x).atan();

            // TODO: Fix panic in `wcs` crate.
            let l1 = lon.as_();
            let l2 = lat.as_();
            let result = std::panic::catch_unwind(|| wcs.proj_lonlat(&wcs::LonLat::new(l1, l2)));

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
                let xr: usize = x.as_();
                let yr: usize = y.as_();

                if (xr > 0) && (xr < image_res_x - 1) && (yr > 0) && (yr < image_res_y - 1) {
                    let cw = w1 * w2;

                    image[(xr, yr)] += cw;
                }
            });

        Ok(ObsImg(image))
    }
}
