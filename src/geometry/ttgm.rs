use bayesfm::geometry::{Geometry, param_value, quaternion_rot};
use nalgebra::{
    ArrayStorage, Const, Dim, Matrix3, RealField, SMatrix, SVector, U6, UnitQuaternion, Vector3,
    VectorView,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate system state type for a torus geometry with arbitrary cross-section.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct XTState<T>
where
    T: RealField,
{
    /// Major torus radius.
    pub rt: T,

    /// Minor torus radius.
    pub rp: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

impl<T> Default for XTState<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            rt: T::zero(),
            rp: T::zero(),
            q: UnitQuaternion::identity(),
        }
    }
}

/// Tapered torus flux rope geometry with elliptical cross-section.
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

impl<T> Geometry<T, 3, 6> for TTGeometry<T>
where
    T: RealField,
{
    const PARAM_NAMES: SVector<&'static str, 6> = SVector::from_array_storage(ArrayStorage(
        [["rot_z", "rot_y", "rot_x", "r_0rs", "d_1au", "cs_delta"]; 1],
    ));

    type CSST = XTState<T>;

    fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
        ics: &VectorView<T, Const<3>, CRStride, CCStride>,
        params: &VectorView<T, Const<6>, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<Matrix3<T>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let major_radius = cs_state.rt.clone();
        let minor_radius = cs_state.rp.clone();
        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let quaternion = cs_state.q.clone();

        let mu = ics[0].clone();
        let nu = ics[1].clone();

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2].clone();
        let psi_half = T::pi() * ics[2].clone();

        let omega = T::two_pi() * nu;
        let com = omega.clone().cos();
        let som = omega.clone().sin();

        let radius_eff = minor_radius.clone() * psi_half.clone().sin().powi(2);
        let denom = (omega.clone().cos().powi(2) + (delta.clone() * omega.sin()).powi(2)).sqrt();

        let dmu = Vector3::from([
            -psi.clone().cos() * com.clone(),
            psi.clone().sin() * com.clone(),
            som.clone(),
        ]) * radius_eff.clone()
            * delta.clone()
            / denom.clone();

        let dnu = Vector3::from([
            psi.clone().cos() * delta.clone().powi(2) * som.clone(),
            -psi.clone().sin() * delta.clone().powi(2) * som.clone(),
            com.clone(),
        ]) * T::two_pi()
            / denom.clone().powi(3)
            * mu.clone()
            * radius_eff
            * delta.clone();

        let ds = Vector3::from([
            -T::two_pi() * mu.clone() * minor_radius.clone() * delta.clone() / denom.clone()
                * (psi_half.clone().sin() * psi_half.clone().cos() * psi.clone().cos()
                    - psi_half.clone().sin().powi(2) * psi.clone().sin())
                * com.clone()
                + T::two_pi() * major_radius.clone() * psi.clone().sin(),
            T::two_pi() * mu.clone() * minor_radius.clone() * delta.clone() / denom.clone()
                * (psi_half.clone().sin() * psi_half.clone().cos() * psi.clone().sin()
                    + psi_half.clone().sin().powi(2) * psi.clone().cos())
                * com
                + T::two_pi() * major_radius * psi.cos(),
            T::two_pi() * mu * minor_radius * delta / denom
                * psi_half.clone().sin()
                * psi_half.cos()
                * som,
        ]);

        Some(SMatrix::from_columns(&[
            quaternion.transform_vector(&dmu),
            quaternion.transform_vector(&dnu),
            quaternion.transform_vector(&ds),
        ]))
    }

    fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
        ics: &VectorView<T, Const<3>, CRStride, CCStride>,
        params: &VectorView<T, U6, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<T>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let major_radius = cs_state.rt.clone();
        let minor_radius = cs_state.rp.clone();
        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let mu = ics[0].clone();
        let nu = ics[1].clone();

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2].clone();
        let psi_half = T::pi() * ics[2].clone();

        let omega = T::two_pi() * nu;

        let radius_eff = minor_radius.clone() * psi_half.sin().powi(2);
        let denom =
            (omega.clone().cos().powi(2) + (delta.clone() * omega.clone().sin()).powi(2)).sqrt();

        let sqrtdetg = mu.clone()
            * minor_radius.clone()
            * delta.clone()
            * ((T::from_usize(2).unwrap() * psi.clone()).cos()
                - (T::from_usize(3).unwrap() * psi.clone()).cos())
            / T::from_usize(2).unwrap()
            * omega.cos()
            + major_radius * denom.clone();

        Some(
            T::two_pi() * sqrtdetg * radius_eff.clone() * delta.clone() * T::two_pi()
                / denom.powi(3)
                * mu
                * radius_eff
                * delta,
        )
    }

    fn initialize_csst<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U6, RStride, CStride>,
        cs_state: &mut Self::CSST,
    ) {
        // Extract parameters using their identifiers.
        let distance_initial = param_value("r_0rs", &Self::PARAM_NAMES, params);
        let diameter_scaled = param_value("d_1au", &Self::PARAM_NAMES, params);
        let longitude = param_value("rot_z", &Self::PARAM_NAMES, params);
        let latitude = param_value("rot_y", &Self::PARAM_NAMES, params);
        let inclination = param_value("rot_x", &Self::PARAM_NAMES, params);

        let rt = distance_initial.clone() * T::from_usize(695510).unwrap()
            / T::from_f64(1.496e8).unwrap();

        assert!(distance_initial > T::zero(), "r_0rs must be positive");
        assert!(diameter_scaled > T::zero(), "d_1au must be positive");

        cs_state.rp = diameter_scaled * rt.clone().powf(T::from_f64(1.14).unwrap())
            / T::from_usize(2).unwrap();
        cs_state.rt = (rt - cs_state.rp.clone()) / T::from_usize(2).unwrap();

        cs_state.q = quaternion_rot(longitude, latitude, inclination);
    }

    fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
        ics: &VectorView<T, Const<3>, CRStride, CCStride>,
        params: &VectorView<T, U6, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let major_radius = cs_state.rt.clone();
        let minor_radius = cs_state.rp.clone();
        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let quaternion = cs_state.q.clone();

        let mu = ics[0].clone();
        let nu = ics[1].clone();

        // Axial coordinate s, but its an angle between [0, 2π].
        let psi = T::two_pi() * ics[2].clone();
        let psi_half = T::pi() * ics[2].clone();

        let omega = T::two_pi() * nu;
        let com = omega.clone().cos();
        let som = omega.clone().sin();

        let radius_eff = minor_radius * psi_half.sin().powi(2);
        let denom = (omega.clone().cos().powi(2) + (delta.clone() * omega.sin()).powi(2)).sqrt();
        let mu_offset = mu * radius_eff * delta / denom;

        Some(quaternion.transform_vector(&Vector3::new(
            major_radius.clone()
                - (major_radius.clone() + mu_offset.clone() * com.clone()) * psi.clone().cos(),
            (major_radius + mu_offset.clone() * com) * psi.sin(),
            mu_offset * som,
        )))
    }

    fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
        ecs: &VectorView<T, Const<3>, CRStride, CCStride>,
        params: &VectorView<T, U6, PRStride, PCStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>>
    where
        CRStride: Dim,
        CCStride: Dim,
        PRStride: Dim,
        PCStride: Dim,
    {
        let major_radius = cs_state.rt.clone();
        let minor_radius = cs_state.rp.clone();
        let quaternion = cs_state.q.clone();

        let delta = param_value("cs_delta", &Self::PARAM_NAMES, params);

        let ecs_norot = quaternion.conjugate().transform_vector(&ecs.clone_owned());

        let x = ecs_norot[0].clone();
        let y = ecs_norot[1].clone();
        let z = ecs_norot[2].clone();

        let psi = if x == major_radius {
            if y > T::zero() {
                T::frac_pi_2()
            } else {
                T::pi() * T::from_f64(1.5).unwrap()
            }
        } else {
            (-y).atan2(x - major_radius.clone()) + T::pi()
        };

        let on_axis = Vector3::new(
            major_radius.clone() - major_radius.clone() * psi.clone().cos(),
            major_radius.clone() * psi.clone().sin(),
            T::zero(),
        );

        let axis_delta = ecs_norot.clone() - on_axis.clone();
        let torus_center = Vector3::new(major_radius, T::zero(), T::zero());

        let dl = if (ecs_norot - torus_center.clone()).norm() >= (on_axis - torus_center).norm() {
            (axis_delta[0].clone().powi(2) + axis_delta[1].clone().powi(2)).sqrt()
        } else {
            -(axis_delta[0].clone().powi(2) + axis_delta[1].clone().powi(2)).sqrt()
        };

        // Compute internal coords (mu, nu).
        let r = (dl.clone().powi(2) + z.clone().powi(2)).sqrt();
        let omega = z.atan2(dl);
        let mut nu = omega.clone() / T::two_pi();

        let radius_eff = minor_radius * (psi.clone() / T::from_usize(2).unwrap()).sin().powi(2);
        let denom = (omega.clone().cos().powi(2) + (delta.clone() * omega.sin()).powi(2)).sqrt();

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
    use bayesfm::geometry::Geometry3D;
    use nalgebra::{SVector, U1, U3, Vector3};

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

        let mut cs_state = XTState::default();

        TTGeometry::initialize_csst(&params.fixed_rows::<6>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.56, 0.17, 0.5);

        let ecs = TTGeometry::transform_internal_to_external::<U1, U3, _, _>(
            &ics_ref.as_view(),
            &params.fixed_rows::<6>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec = TTGeometry::transform_external_to_internal::<U1, U3, _, _>(
            &ecs.as_view(),
            &params.fixed_rows::<6>(0),
            &cs_state,
        )
        .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        TTGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<6>(0), 1e-6);
    }
}
