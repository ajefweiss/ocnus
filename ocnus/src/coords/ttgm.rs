use crate::{
    coords::{CoordsError, OcnusCoords},
    fXX,
    math::{T, atan2, cos, powf, powi, quaternion_rot, sin, sqrt},
};
use nalgebra::{Const, Dim, U1, UnitQuaternion, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate system cs_state type for a tapered torus geometry with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTState<T>
where
    T: fXX,
{
    /// Major torus radius
    pub major_radius: T,

    /// Minor torus radius
    pub minor_radius: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

/// Tapered torus flux rope geometry.
pub struct TTGeometry<T>(PhantomData<T>)
where
    T: fXX;

impl<T> Default for TTGeometry<T>
where
    T: fXX,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusCoords<T, 6, TTState<T>> for TTGeometry<T>
where
    T: fXX,
{
    const PARAMS: [&'static str; 6] = [
        "longitude",
        "latitude",
        "inclination",
        "distance_0",
        "diameter_1au",
        "delta",
    ];

    fn contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        cs_state: &TTState<T>,
    ) -> Result<[Vector3<T>; 3], CoordsError> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = Self::param_value("delta", params).unwrap();

        let quaternion = cs_state.q;

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;
        let com = cos!(omega);
        let som = sin!(omega);

        let radius_eff = minor_radius * powi!(sin!(psi_half), 2);
        let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

        let dmu =
            Vector3::from([-cos!(psi) * com, sin!(psi) * com, som]) * radius_eff * delta / denom;

        let dnu = Vector3::from([
            cos!(psi) * powi!(delta, 2) * som,
            -sin!(psi) * powi!(delta, 2) * som,
            com,
        ]) * T::two_pi()
            / powf!(denom, T!(3.0))
            * mu
            * radius_eff
            * delta;

        let ds = Vector3::from([
            -T::two_pi() * mu * minor_radius * delta / denom
                * (sin!(psi_half) * cos!(psi_half) * cos!(psi)
                    - powi!(sin!(psi_half), 2) * sin!(psi))
                * com
                + T::two_pi() * major_radius * sin!(psi),
            T::two_pi() * mu * minor_radius * delta / denom
                * (sin!(psi_half) * cos!(psi_half) * sin!(psi)
                    + powi!(sin!(psi_half), 2) * cos!(psi))
                * com
                + T::two_pi() * major_radius * cos!(psi),
            T::two_pi() * mu * minor_radius * delta / denom * sin!(psi_half) * cos!(psi_half) * som,
        ]);

        Ok([
            quaternion.transform_vector(&dmu),
            quaternion.transform_vector(&dnu),
            quaternion.transform_vector(&ds),
        ])
    }

    /// Compute the determinant of the metric tensor.
    fn detg<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        cs_state: &TTState<T>,
    ) -> Result<T, CoordsError> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = Self::param_value("delta", params).unwrap();

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;

        let radius_eff = minor_radius * powi!(sin!(psi_half), 2);
        let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

        let detg = mu * minor_radius * delta * (cos!(T!(2.0) * psi) - cos!(T!(3.0) * psi))
            / T!(2.0)
            * cos!(omega)
            + major_radius * denom;

        Ok(
            T::two_pi() * detg * radius_eff * delta * T::two_pi() / powf!(denom, T!(3.0))
                * mu
                * radius_eff
                * delta,
        )
    }

    fn initialize_cs<CStride: Dim>(
        params: &VectorView<T, Const<6>, U1, CStride>,
        cs_state: &mut TTState<T>,
    ) -> Result<(), CoordsError> {
        // Extract parameters using their identifiers.
        let distance_0 = Self::param_value("distance_0", params).unwrap() * T!(695510.0);
        let diameter_1au = Self::param_value("diameter_1au", params).unwrap();
        let longitude = Self::param_value("longitude", params).unwrap();
        let latitude = Self::param_value("latitude", params).unwrap();
        let inclination = Self::param_value("inclination", params).unwrap();

        let rt = distance_0 / T!(1.496e8);

        cs_state.minor_radius = diameter_1au * powf!(rt, T!(1.14)) / T!(2.0);
        cs_state.major_radius = (rt - cs_state.minor_radius) / T!(2.0);

        cs_state.q = quaternion_rot(longitude, latitude, inclination);

        Ok(())
    }

    fn transform_ics_to_ecs<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        cs_state: &TTState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let delta = Self::param_value("delta", params).unwrap();

        let quaternion = cs_state.q;

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2];
        let psi_half = T::pi() * ics[2];

        let omega = T::two_pi() * nu;
        let com = cos!(omega);
        let som = sin!(omega);

        let radius_eff = minor_radius * powi!(sin!(psi_half), 2);
        let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));
        let mu_offset = mu * radius_eff * delta / denom;

        Ok(quaternion.transform_vector(&Vector3::new(
            major_radius - (major_radius + mu_offset * com) * cos!(psi),
            (major_radius + mu_offset * com) * sin!(psi),
            mu_offset * som,
        )))
    }

    fn transform_ecs_to_ics<CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        cs_state: &TTState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let major_radius = cs_state.major_radius;
        let minor_radius = cs_state.minor_radius;
        let quaternion = cs_state.q;

        let delta = Self::param_value("delta", params).unwrap();

        let ecs_norot = quaternion.conjugate().transform_vector(&ecs.clone_owned());

        let x = ecs_norot[0];
        let y = ecs_norot[1];
        let z = ecs_norot[2];

        let psi = if x == major_radius {
            if y > T::zero() {
                T::pi() / T!(2.0)
            } else {
                T::pi() * T!(1.5)
            }
        } else {
            atan2!(-y, x - major_radius) + T::pi()
        };

        let on_axis = Vector3::new(
            major_radius - major_radius * cos!(psi),
            major_radius * sin!(psi),
            T::zero(),
        );

        let axis_delta = ecs_norot - on_axis;
        let torus_center = Vector3::new(major_radius, T::zero(), T::zero());

        let dl = if (ecs_norot - torus_center).norm() >= (on_axis - torus_center).norm() {
            sqrt!(powi!(axis_delta[0], 2) + powi!(axis_delta[1], 2))
        } else {
            -sqrt!(powi!(axis_delta[0], 2) + powi!(axis_delta[1], 2))
        };

        // Compute internal coordinates (mu, nu).
        let r = sqrt!(powi!(dl, 2) + powi!(z, 2));
        let omega = atan2!(z, dl);
        let mut nu = omega / T::two_pi();

        let radius_eff = minor_radius * powi!(sin!(psi / T!(2.0)), 2);
        let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

        let mu = r / delta / radius_eff * denom;

        // Force nu into [0, 1].
        while nu < T::zero() {
            nu += T::one()
        }

        while nu > T::one() {
            nu -= T::one()
        }

        Ok(Vector3::new(mu, nu, psi / T::two_pi()))
    }
}

#[cfg(test)]
mod tests {
    use crate::coords::{OcnusCoords, TTGeometry, TTState};
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

        TTGeometry::initialize_cs(&params.fixed_rows::<6>(0), &mut cs_state).unwrap();

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

        assert!((ics_rec - ics_ref).norm() < 1e-4);

        TTGeometry::test_contravariant_basis(&ics_ref.as_view(), &params.fixed_rows::<6>(0), 1e-5);
    }
}
