use crate::{geom::OcnusGeometry, math::quaternion_xyz, t_from};
use nalgebra::{Const, Dim, RealField, SimdRealField, U1, Vector3, VectorView, VectorView3};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Model state for a tapered torus with arbitrary croFloat::sin(psi)-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TTState<T> {
    /// Major torus radius
    pub major_radius: T,

    /// Minor torus radius
    pub minor_radius: T,
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

        Vector3::new(
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
        )
    }

    fn coords_xyz_to_ics<CStride: Dim>(
        xyz: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> Vector3<T> {
        let major_radius = geom_state.major_radius;
        let minor_radius = geom_state.minor_radius;
        let delta = Self::param_value("delta", params);

        unimplemented!()
    }

    fn create_xyz_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        ics_comp: &VectorView3<T>,
        params: &VectorView<T, Const<6>, U1, CStride>,
        geom_state: &TTState<T>,
    ) -> Vector3<T> {
    }
}
