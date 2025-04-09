use crate::{geom::OcnusGeometry, math::quaternion_xyz, t_from};
use nalgebra::{
    Const, Dim, RealField, Scalar, SimdRealField, U1, UnitQuaternion, Vector3, VectorView,
    VectorView3,
};
use num_traits::{Float, Zero};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Model state for a tapered torus with arbitrary croFloat::sin(psi)-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTState<T>
where
    T: Clone + RealField + Scalar + Zero,
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
    T: Float + RealField + SimdRealField;

impl<T> Default for TTGeometry<T>
where
    T: Float + RealField + SimdRealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusGeometry<T, 6, TTState<T>> for TTGeometry<T>
where
    T: Float + RealField + SimdRealField,
{
    const PARAMS: [&'static str; 6] = [
        "longitude",
        "latitude",
        "inclination",
        "major_radius_0",
        "minor_radius_0",
        "delta",
    ];

    fn basis_vectors<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> [Vector3<T>; 3] {
        let major_radius = geom_state.major_radius;
        let minor_radius = geom_state.minor_radius;
        let delta = Self::param_value("delta", params);

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let tpi = t_from!(2.0) * T::pi();
        let psi = tpi * ics[2];
        let psi2 = T::pi() * ics[2];

        let omega = tpi * nu;
        let com = Float::cos(omega);
        let som = Float::sin(omega);

        let radius_eff = minor_radius * Float::powi(Float::sin(psi2), 2);

        let denom = Float::sqrt(
            Float::powi(Float::cos(omega), 2) + Float::powi(delta * Float::sin(omega), 2),
        );

        let dmu = Vector3::from([
            radius_eff * delta / denom * Float::cos(psi) * com,
            radius_eff * delta / denom * Float::sin(psi) * com,
            radius_eff * delta / denom * som,
        ]);

        let cos_factor =
            (-t_from!(2.0 * Float::sqrt(2.0)) * Float::powi(delta, 2) * tpi * Float::sin(tpi * nu))
                / Float::powf(
                    T::one() + Float::powi(delta, 2) + Float::cos(t_from!(2.0) * tpi * nu)
                        - Float::powi(delta, 2) * Float::cos(t_from!(2.0) * tpi * nu),
                    t_from!(1.5),
                );

        let sin_factor = (t_from!(2.0 * Float::sqrt(2.0)) * tpi * Float::cos(tpi * nu))
            / Float::powf(
                T::one() + Float::powi(delta, 2) + Float::cos(t_from!(2.0) * tpi * nu)
                    - Float::powi(delta, 2) * Float::cos(t_from!(2.0) * tpi * nu),
                t_from!(1.5),
            );

        let dnu = Vector3::from([
            mu * radius_eff
                * delta
                * Float::powi(Float::sin(psi2), 2)
                * Float::cos(psi)
                * cos_factor,
            mu * radius_eff
                * delta
                * Float::powi(Float::sin(psi2), 2)
                * Float::sin(psi)
                * cos_factor,
            mu * radius_eff * delta * Float::powi(Float::sin(psi2), 2) * sin_factor,
        ]);

        let ds = Vector3::from([
            tpi * major_radius * Float::sin(psi)
                + mu * radius_eff * delta / denom
                    * tpi
                    * (Float::sin(psi2) * Float::cos(psi2) * Float::cos(psi)
                        - Float::powi(Float::sin(psi2), 2) * Float::sin(psi))
                    * com,
            tpi * major_radius * Float::cos(psi)
                + mu * radius_eff * delta / denom
                    * tpi
                    * (Float::sin(psi2) * Float::cos(psi2) * Float::sin(psi)
                        + Float::powi(Float::sin(psi2), 2) * Float::cos(psi))
                    * som,
            tpi * mu * radius_eff * delta / denom
                * Float::sin(psi / t_from!(2.0))
                * Float::cos(psi / t_from!(2.0))
                * som,
        ]);

        [dmu, dnu, ds]
    }

    fn coords_ics_to_xyz<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> Vector3<T> {
        let major_radius = geom_state.major_radius;
        let minor_radius = geom_state.minor_radius;
        let quaternion = geom_state.q;

        let delta = Self::param_value("delta", params);

        let mu = ics[0];
        let nu = ics[1];

        // Axial coordinate s, but its an angle between [0, 2π].
        let tpi = t_from!(2.0) * T::pi();
        let psi = tpi * ics[2];
        let psi2 = T::pi() * ics[2];

        let omega = tpi * nu;
        let com = Float::cos(omega);
        let som = Float::sin(omega);

        let radius_eff = minor_radius * Float::powi(Float::sin(psi2), 2);

        let denom = Float::sqrt(
            Float::powi(Float::cos(omega), 2) + Float::powi(delta * Float::sin(omega), 2),
        );

        let xyz = Vector3::new(
            major_radius - major_radius * Float::cos(psi)
                + mu * radius_eff * delta / denom
                    * Float::powi(Float::sin(psi2), 2)
                    * Float::cos(psi)
                    * com,
            major_radius * Float::sin(psi)
                + mu * radius_eff * delta / denom
                    * Float::powi(Float::sin(psi2), 2)
                    * Float::sin(psi)
                    * com,
            mu * radius_eff * delta / denom * Float::powi(Float::sin(psi2), 2) * som,
        );

        quaternion.transform_vector(&xyz)
    }

    fn coords_xyz_to_ics<CStride: Dim>(
        xyz: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> Vector3<T> {
        let major_radius = geom_state.major_radius;
        let minor_radius = geom_state.minor_radius;
        let quaternion = geom_state.q;

        let delta = Self::param_value("delta", params);

        let xyz_derot = quaternion.conjugate().transform_vector(&xyz.clone_owned());

        let x = xyz_derot[0];
        let y = xyz_derot[1];

        let psi = if x == major_radius {
            if y > T::zero() {
                T::pi() / t_from!(2.0)
            } else {
                T::pi() * t_from!(1.5)
            }
        } else {
            Float::atan2(-y, x - major_radius) + T::pi()
        };

        let axis_proj = Vector3::new(
            major_radius - major_radius * Float::cos(psi),
            major_radius * Float::sin(psi),
            T::zero(),
        );

        let xyz_delta = xyz - axis_proj;

        let dl = Float::sqrt(Float::powi(xyz_delta[0], 2) + Float::powi(xyz_delta[1], 2));
        let z = xyz_delta[2];

        // Compute intrinsic coordinates (mu, nu).
        let r = Float::sqrt(Float::powi(dl, 2) + Float::powi(z, 2));
        let mu = r * Float::sqrt(Float::powi(dl, 2) + Float::powi(z, 2) * Float::powi(delta, 2))
            / r
            / delta
            / minor_radius;
        let nu = psi - Float::acos(dl / r) / t_from!(2.0) * T::pi();

        Vector3::new(mu, nu, psi)
    }

    fn create_xyz_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        vec: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> Vector3<T> {
        let quaternion = geom_state.q;

        let [d1, d2, d3] = Self::basis_vectors(ics, params, geom_state);

        let vec = d1 * vec[0] + d2 * vec[1] + d3 * vec[2];

        quaternion.transform_vector(&vec)
    }

    fn geom_state<CStride: Dim>(
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &mut TTState<T>,
    ) {
        let longitude = Self::param_value("longitude", params);
        let latitude = Self::param_value("latitude", params);
        let inclination = Self::param_value("inclination", params);

        geom_state.major_radius = Self::param_value("major_radius_0", params);
        geom_state.minor_radius = Self::param_value("minor_radius_0", params);

        geom_state.q = quaternion_xyz(longitude, latitude, inclination);
    }
}

#[cfg(test)]
mod tests {
    use super::{TTGeometry, TTState};
    use crate::geom::OcnusGeometry;
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_tt_cords() {
        let params = SVector::<f64, 6>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            0.0,
            0.5,
            0.1,
            1.0,
        ]);

        let state = TTState::default();

        let ics_coords = Vector3::new(0.5, 0.1, 0.5);

        let xyz = TTGeometry::coords_ics_to_xyz(
            &ics_coords.as_view(),
            &params.fixed_rows::<6>(0),
            &state,
        );

        println!("{}", xyz);
    }
}
