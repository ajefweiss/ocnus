use crate::coords::{OcnusCoords, param_value, quaternion_rot};
use nalgebra::{
    ArrayStorage, Dim, RealField, SVector, U6, UnitQuaternion, Vector3, VectorView, VectorView3,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate system state type for a torus geometry with arbitrary cross-section.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTState<T>
where
    T: RealField,
{
    /// Major torus radius
    pub major_radius: T,

    /// Minor torus radius
    pub minor_radius: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

/// Tapered torus flux rope geometry wit elliptical cross-section.
pub struct TTGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for TTGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusCoords<T, 6, TTState<T>> for TTGeometry<T>
where
    T: Copy + RealField,
{
    const PARAMS: SVector<&'static str, 6> = SVector::from_array_storage(ArrayStorage(
        [[
            "longitude",
            "latitude",
            "inclination",
            "distance_0",
            "diameter_1au",
            "delta",
        ]; 1],
    ));

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &TTState<T>,
    ) -> Option<[Vector3<T>; 3]> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = param_value("delta", &Self::PARAMS, params).unwrap();

        let quaternion = cs_state.q;

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;
        let com = omega.cos();
        let som = omega.sin();

        let radius_eff = minor_radius * psi_half.sin().powi(2);
        let denom = (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();

        let dmu =
            Vector3::from([-psi.cos() * com, psi.sin() * com, som]) * radius_eff * delta / denom;

        let dnu = Vector3::from([
            psi.cos() * delta.powi(2) * som,
            -psi.sin() * delta.powi(2) * som,
            com,
        ]) * T::two_pi()
            / denom.powi(3)
            * mu
            * radius_eff
            * delta;

        let ds = Vector3::from([
            -T::two_pi() * mu * minor_radius * delta / denom
                * (psi_half.sin() * psi_half.cos() * psi.cos()
                    - psi_half.sin().powi(2) * psi.sin())
                * com
                + T::two_pi() * major_radius * psi.sin(),
            T::two_pi() * mu * minor_radius * delta / denom
                * (psi_half.sin() * psi_half.cos() * psi.sin()
                    + psi_half.sin().powi(2) * psi.cos())
                * com
                + T::two_pi() * major_radius * psi.cos(),
            T::two_pi() * mu * minor_radius * delta / denom * psi_half.sin() * psi_half.cos() * som,
        ]);

        Some([
            quaternion.transform_vector(&dmu),
            quaternion.transform_vector(&dnu),
            quaternion.transform_vector(&ds),
        ])
    }

    /// Compute the determinant of the metric tensor.
    fn detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &TTState<T>,
    ) -> Option<T> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = param_value("delta", &Self::PARAMS, params).unwrap();

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;

        let radius_eff = minor_radius * psi_half.sin().powi(2);
        let denom = (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();

        let detg = mu
            * minor_radius
            * delta
            * ((T::from_usize(2).unwrap() * psi).cos() - (T::from_usize(3).unwrap() * psi).cos())
            / T::from_usize(2).unwrap()
            * omega.cos()
            + major_radius * denom;

        Some(
            T::two_pi() * detg * radius_eff * delta * T::two_pi() / denom.powi(3)
                * mu
                * radius_eff
                * delta,
        )
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &mut TTState<T>,
    ) {
        // Extract parameters using their identifiers.
        let distance_0 = param_value("distance_0", &Self::PARAMS, params).unwrap()
            * T::from_usize(695510).unwrap();
        let diameter_1au = param_value("diameter_1au", &Self::PARAMS, params).unwrap();
        let longitude = param_value("longitude", &Self::PARAMS, params).unwrap();
        let latitude = param_value("latitude", &Self::PARAMS, params).unwrap();
        let inclination = param_value("inclination", &Self::PARAMS, params).unwrap();

        let rt = distance_0 / T::from_f64(1.496e8).unwrap();

        assert!(distance_0 > T::zero(), "initial distance must be positive");
        assert!(diameter_1au > T::zero(), "diameter must be positive");

        cs_state.minor_radius =
            diameter_1au * rt.powf(T::from_f64(1.14).unwrap()) / T::from_usize(2).unwrap();
        cs_state.major_radius = (rt - cs_state.minor_radius) / T::from_usize(2).unwrap();

        cs_state.q = quaternion_rot(longitude, latitude, inclination);
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &TTState<T>,
    ) -> Option<Vector3<T>> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = param_value("delta", &Self::PARAMS, params).unwrap();

        let quaternion = cs_state.q;

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;
        let com = omega.cos();
        let som = omega.sin();

        let radius_eff = minor_radius * psi_half.sin().powi(2);
        let denom = (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();
        let mu_offset = mu * radius_eff * delta / denom;

        Some(quaternion.transform_vector(&Vector3::new(
            major_radius - (major_radius + mu_offset * com) * psi.cos(),
            (major_radius + mu_offset * com) * psi.sin(),
            mu_offset * som,
        )))
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &TTState<T>,
    ) -> Option<Vector3<T>> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let quaternion = cs_state.q;

        let delta = param_value("delta", &Self::PARAMS, params).unwrap();

        let ecs_norot = quaternion.conjugate().transform_vector(&ecs.clone_owned());

        let x = ecs_norot[0];
        let y = ecs_norot[1];
        let z = ecs_norot[2];

        let psi = if x == major_radius {
            if y > T::zero() {
                T::frac_pi_2()
            } else {
                T::pi() * T::from_f64(1.5).unwrap()
            }
        } else {
            (-y).atan2(x - major_radius) + T::pi()
        };

        let on_axis = Vector3::new(
            major_radius - major_radius * psi.cos(),
            major_radius * psi.sin(),
            T::zero(),
        );

        let axis_delta = ecs_norot - on_axis;
        let torus_center = Vector3::new(major_radius, T::zero(), T::zero());

        let dl = if (ecs_norot - torus_center).norm() >= (on_axis - torus_center).norm() {
            (axis_delta[0].powi(2) + axis_delta[1].powi(2)).sqrt()
        } else {
            -(axis_delta[0].powi(2) + axis_delta[1].powi(2)).sqrt()
        };

        // Compute internal coordinates (mu, nu).
        let r = (dl.powi(2) + z.powi(2)).sqrt();
        let omega = z.atan2(dl);
        let mut nu = omega / T::two_pi();

        let radius_eff = minor_radius * (psi / T::from_usize(2).unwrap()).sin().powi(2);
        let denom = (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();

        let mu = r / delta / radius_eff * denom;

        // Force nu into [0, 1].
        while nu < T::zero() {
            nu += T::one()
        }

        while nu > T::one() {
            nu -= T::one()
        }

        Some(Vector3::new(mu, nu, psi / T::two_pi()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_tt_coords() {
        let params = SVector::<f64, 6>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            7.0_f64.to_radians(),
            0.5,
            0.1,
            0.8,
        ]);

        let mut cs_state = TTState::default();

        TTGeometry::initialize_cs(&params.fixed_rows::<6>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.56, 0.17, 0.5);

        let ecs = TTGeometry::transform_ics_to_ecs(
            &ics_ref.as_view(),
            &params.fixed_rows::<6>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec =
            TTGeometry::transform_ecs_to_ics(&ecs.as_view(), &params.fixed_rows::<6>(0), &cs_state)
                .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        TTGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<6>(0), 1e-6);
    }
}
