use crate::{
    coords::{CoordsError, OcnusCoords},
    fXX,
    math::{T, acos, atan2, cos, powf, powi, quaternion_rot, sin, sqrt},
};
use nalgebra::{Const, Dim, U1, UnitQuaternion, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate state type for a tapered torus geometry with arbitrary cross-section shapes.
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

/// Tapered torus geometry.
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
    const COORD_PARAMS: [&'static str; 6] = [
        "longitude",
        "latitude",
        "inclination",
        "major_radius_0",
        "minor_radius_0",
        "delta",
    ];

    fn contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        state: &TTState<T>,
    ) -> Result<[Vector3<T>; 3], CoordsError> {
        let major_radius = state.major_radius;
        let minor_radius = state.minor_radius;
        let delta = Self::param_value("delta", params).unwrap();

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

        let dmu = Vector3::from([
            radius_eff * delta / denom * cos!(psi) * com,
            radius_eff * delta / denom * sin!(psi) * com,
            radius_eff * delta / denom * som,
        ]);

        let cos_factor = (-T!(sqrt!(8.0)) * powi!(delta, 2) * T::two_pi() * sin!(T::two_pi() * nu))
            / powf!(
                T::one() + powi!(delta, 2) + cos!(T::four_pi() * nu)
                    - powi!(delta, 2) * cos!(T::four_pi() * nu),
                T!(1.5)
            );

        let sin_factor = (T!(sqrt!(8.0)) * T::two_pi() * cos!(T::two_pi() * nu))
            / powf!(
                T::one() + powi!(delta, 2) + cos!(T::four_pi() * nu)
                    - powi!(delta, 2) * cos!(T::four_pi() * nu),
                T!(1.5)
            );

        let dnu = Vector3::from([
            mu * radius_eff * delta * powi!(sin!(psi_half), 2) * cos!(psi) * cos_factor,
            mu * radius_eff * delta * powi!(sin!(psi_half), 2) * sin!(psi) * cos_factor,
            mu * radius_eff * delta * powi!(sin!(psi_half), 2) * sin_factor,
        ]);

        let ds = Vector3::from([
            T::two_pi() * major_radius * sin!(psi)
                + mu * radius_eff * delta / denom
                    * T::two_pi()
                    * (sin!(psi_half) * cos!(psi_half) * cos!(psi)
                        - powi!(sin!(psi_half), 2) * sin!(psi))
                    * com,
            T::two_pi() * major_radius * cos!(psi)
                + mu * radius_eff * delta / denom
                    * T::two_pi()
                    * (sin!(psi_half) * cos!(psi_half) * sin!(psi)
                        + powi!(sin!(psi_half), 2) * cos!(psi))
                    * som,
            T::two_pi() * mu * radius_eff * delta / denom * sin!(psi_half) * cos!(psi_half) * som,
        ]);

        Ok([dmu, dnu, ds])
    }

    fn transform_ics_to_ecs<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        state: &TTState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let major_radius = state.major_radius;
        let minor_radius = state.minor_radius;
        let quaternion = state.q;

        let delta = Self::param_value("delta", params).unwrap();

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
        state: &TTState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let major_radius = state.major_radius;
        let minor_radius = state.minor_radius;
        let quaternion = state.q;

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

        let dl = sqrt!(powi!(axis_delta[0], 2) + powi!(axis_delta[1], 2));

        // Compute internal coordinates (mu, nu).
        let r = sqrt!(powi!(dl, 2) + powi!(z, 2));
        let omega = acos!(dl / r);
        let nu = omega / T::two_pi();

        let radius_eff = minor_radius * powi!(sin!(psi / T!(2.0)), 2);
        let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

        let mu = r / delta / radius_eff * denom;

        Ok(Vector3::new(mu, nu, psi / T::two_pi()))
    }

    fn initialize_cst<CStride: Dim>(
        params: &VectorView<T, Const<6>, U1, CStride>,
        state: &mut TTState<T>,
    ) {
        let longitude = Self::param_value("longitude", params).unwrap();
        let latitude = Self::param_value("latitude", params).unwrap();
        let inclination = Self::param_value("inclination", params).unwrap();

        state.major_radius = Self::param_value("major_radius_0", params).unwrap();
        state.minor_radius = Self::param_value("minor_radius_0", params).unwrap();

        state.q = quaternion_rot(longitude, latitude, inclination);
    }
}

#[cfg(test)]
mod tests {
    use crate::coords::OcnusCoords;

    use super::{TTGeometry, TTState};
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_tt_basis() {
        let params = SVector::<f64, 6>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            7.0_f64.to_radians(),
            0.5,
            0.1,
            0.9,
        ]);

        let mut state = TTState::default();

        TTGeometry::initialize_cst(&params.fixed_rows::<6>(0), &mut state);

        let ics = Vector3::new(0.56, 0.17, 0.45);

        let ics_1p = Vector3::new(0.56001, 0.17, 0.45);
        let ics_1m = Vector3::new(0.55999, 0.17, 0.45);

        let ics_2p = Vector3::new(0.56, 0.17001, 0.45);
        let ics_2m = Vector3::new(0.56, 0.16999, 0.45);

        let ics_3p = Vector3::new(0.56, 0.17, 0.45001);
        let ics_3m = Vector3::new(0.56, 0.17, 0.4499);

        let basis =
            TTGeometry::contravariant_basis(&ics.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_1p =
            TTGeometry::transform_ics_to_ecs(&ics_1p.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_1m =
            TTGeometry::transform_ics_to_ecs(&ics_1m.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_2p =
            TTGeometry::transform_ics_to_ecs(&ics_2p.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_2m =
            TTGeometry::transform_ics_to_ecs(&ics_2m.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_3p =
            TTGeometry::transform_ics_to_ecs(&ics_3p.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        let ecs_3m =
            TTGeometry::transform_ics_to_ecs(&ics_3m.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        dbg!(basis[0]);
        dbg!(&(ecs_1p - ecs_1m) / 0.001);

        assert!((basis[0] - (ecs_1p - ecs_1m) / 0.001).norm() < 1e-4);
        assert!((basis[1] - (ecs_2p - ecs_2m) / 0.001).norm() < 1e-4);
        assert!((basis[2] - (ecs_3p - ecs_3m) / 0.001).norm() < 1e-4);
    }

    #[test]
    fn test_tt_coords() {
        let params = SVector::<f64, 6>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            7.0_f64.to_radians(),
            0.5,
            0.1,
            0.9,
        ]);

        let mut state = TTState::default();

        TTGeometry::initialize_cst(&params.fixed_rows::<6>(0), &mut state);

        let ics_ref = Vector3::new(0.56, 0.17, 0.45);

        let ecs = TTGeometry::transform_ics_to_ecs(
            &ics_ref.as_view(),
            &params.fixed_rows::<6>(0),
            &state,
        )
        .unwrap();

        let ics_rec =
            TTGeometry::transform_ecs_to_ics(&ecs.as_view(), &params.fixed_rows::<6>(0), &state)
                .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-4);
    }
}
