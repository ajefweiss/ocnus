use crate::coords::{OcnusCoords, param_value, quaternion_rot};
use nalgebra::{
    ArrayStorage, Const, DefaultAllocator, Dim, DimName, OVector, RealField, UnitQuaternion,
    Vector3, VectorView, VectorView3, allocator::Allocator,
};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate system state type for cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCState<T>
where
    T: Copy + RealField,
{
    /// Offset along the x-axis.
    pub x: T,

    /// Offset along the z-axis.
    pub z: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

/// The circular-cylindric contravariant basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_basis<T, D, RStride, CStride>(
    (r, nu, _z): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let omega = T::two_pi() * nu;
    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let dr = Vector3::from([omega.cos(), T::zero(), omega.sin()]) * radius_linearized;
    let dnu =
        Vector3::from([-omega.sin(), T::zero(), omega.cos()]) * T::two_pi() * r * radius_linearized;
    let dz = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dr, dnu, dz]
}

/// The elliptic-cylindric contravariant basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_basis<T, D, RStride, CStride>(
    (mu, nu, _s): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let delta = param_value("delta", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let omega = T::two_pi() * nu;
    let com = omega.cos();
    let som = omega.sin();

    let denom = (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();

    let nu_nom = T::two_pi() * delta * mu * radius_linearized;

    let dmu = Vector3::from([
        delta * radius_linearized * com / denom,
        T::zero(),
        delta * radius_linearized * som / denom,
    ]);

    let dnu = Vector3::from([
        -nu_nom * delta.powi(2) * som / denom.powi(3),
        T::zero(),
        nu_nom * com / denom.powi(3),
    ]);

    let ds = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dmu, dnu, ds]
}

/// The circular-cylindric metric determinant.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_detg<T, D, RStride, CStride>(
    (r, _nu, _z): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> T
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    T::two_pi() * r * radius_linearized.powi(2)
}

/// The elliptic-cylindric metric determinant
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_detg<T, D, RStride, CStride>(
    (mu, _nu, _s): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> T
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let delta = param_value("delta", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    T::two_pi() * mu * delta.powi(2) * radius_linearized.powi(2)
}

/// The circular-cylindric coordinate transformation (ecs -> ics).
pub fn cc_ecs_to_ics<T, D, RStride, CStride>(
    (x, y, z): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let radius = param_value("radius", names, params).unwrap();
    let y_offset = param_value("y", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    // Compute polar coordinates (r, omega).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;

    let mut nu = if r == T::zero() {
        T::zero()
    } else {
        z.atan2(x) / T::two_pi()
    };

    // Force nu into [0, 1].
    while nu < T::zero() {
        nu += T::one()
    }

    while nu > T::one() {
        nu -= T::one()
    }

    Vector3::new(r, nu, y)
}

/// The elliptic-cylindric coordinate transformation (ecs -> ics).
pub fn ec_ecs_to_ics<T, D, RStride, CStride>(
    (x, y, z): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let delta = param_value("delta", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    // Compute internal coordinates (mu, nu).
    let r = (x.powi(2) + z.powi(2)).sqrt();

    let (mu, mut nu) = if r == T::zero() {
        (T::zero(), T::zero())
    } else {
        (
            r * (x.powi(2) + z.powi(2) * delta.powi(2)).sqrt() / r / delta / radius_linearized,
            z.atan2(x) / T::from_usize(2).unwrap() / T::pi(),
        )
    };

    // Force nu into [0, 1].
    while nu < T::zero() {
        nu += T::one()
    }

    while nu > T::one() {
        nu -= T::one()
    }

    Vector3::new(mu, nu, y)
}

/// The circular-cylindric coordinate transformation (ics -> ecs).
pub fn cc_ics_to_ecs<T, D, RStride, CStride>(
    (r, nu, y): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let radius = param_value("radius", names, params).unwrap();
    let y_offset = param_value("y", names, params).unwrap();

    let omega = T::two_pi() * nu;
    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = omega.cos() * r * radius_linearized;
    let z = omega.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

/// The elliptic-cylindric coordinate transformation (ics -> ecs).
pub fn ec_ics_to_ecs<T, D, RStride, CStride>(
    (mu, nu, s): (T, T, T),
    names: &OVector<&'static str, D>,
    params: &VectorView<T, D, RStride, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    T: Copy + RealField,
    D: DimName,
    DefaultAllocator: Allocator<D>,
    RStride: Dim,
    CStride: Dim,
{
    let y_offset = param_value("y", names, params).unwrap();
    let delta = param_value("delta", names, params).unwrap();
    let radius = param_value("radius", names, params).unwrap();

    let radius_linearized = radius * (T::one() - y_offset.powi(2)).sqrt();

    let omega = T::two_pi() * nu;

    let df = mu * delta * radius_linearized
        / (omega.cos().powi(2) + (delta * omega.sin()).powi(2)).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = omega.cos() * df;
    let y = omega.sin() * df;

    Vector3::new(x, s, y)
}

macro_rules! impl_xcgm_geom {
    ($model: ident, $docs: literal, $params: expr, $fn_basis: tt, $fn_detg:tt, $fn_ics: tt, $fn_ecs: tt) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        #[derive(Debug)]
        pub struct $model<T>(PhantomData<T>)
        where
            T: RealField;

        impl<T> Default for $model<T>
        where
            T: Copy + RealField,
        {
            fn default() -> Self {
                Self(PhantomData::<T>)
            }
        }

        impl<T> OcnusCoords<T, Const<{ $params.len() }>, XCState<T>> for $model<T>
        where
            T: Copy + RealField,
        {
            const PARAMS: OVector<&'static str, Const<{ $params.len() }>> =
                OVector::from_array_storage(ArrayStorage([$params; 1]));

            fn contravariant_basis<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &XCState<T>,
            ) -> Option<[Vector3<T>; 3]>
            where
                RStride: Dim,
                CStride: Dim,
            {
                let quaternion = cs_state.q;

                let [dmu, dnu, ds] = $fn_basis::<T, Const<{ $params.len() }>, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                );

                Some([
                    quaternion.transform_vector(&dmu),
                    quaternion.transform_vector(&dnu),
                    quaternion.transform_vector(&ds),
                ])
            }

            /// Compute the determinant of the metric tensor.
            fn detg<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &XCState<T>,
            ) -> Option<T>
            where
                RStride: Dim,
                CStride: Dim,
            {
                Some($fn_detg::<T, Const<{ $params.len() }>, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                ))
            }

            fn initialize_cs<RStride, CStride>(
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &mut XCState<T>,
            ) where
                RStride: Dim,
                CStride: Dim,
            {
                let phi = param_value("phi", &Self::PARAMS, params).unwrap();
                let theta = param_value("theta", &Self::PARAMS, params).unwrap();
                let psi = param_value("psi", &Self::PARAMS, params).unwrap_or(T::zero());
                let radius = param_value("radius", &Self::PARAMS, params).unwrap();
                let x_init = param_value("x_0", &Self::PARAMS, params).unwrap();
                let y = param_value("y", &Self::PARAMS, params).unwrap();

                assert!(radius > T::zero(), "radius must be positive");

                cs_state.x = x_init;
                cs_state.z = radius * y * (T::one() - (phi.sin() * theta.sin()).powi(2)).sqrt()
                    / phi.cos()
                    / theta.cos();

                cs_state.q = quaternion_rot(phi, psi, theta);
            }

            fn transform_ics_to_ecs<RStride, CStride>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &XCState<T>,
            ) -> Option<Vector3<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                let quaternion = cs_state.q;

                let ecs_norot = $fn_ecs::<T, Const<{ $params.len() }>, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                );

                Some(
                    quaternion.transform_vector(&ecs_norot)
                        + Vector3::new(cs_state.x, T::zero(), cs_state.z),
                )
            }

            fn transform_ecs_to_ics<RStride, CStride>(
                ecs: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &XCState<T>,
            ) -> Option<Vector3<T>>
            where
                RStride: Dim,
                CStride: Dim,
            {
                let quaternion = cs_state.q;

                let ecs_norot = quaternion
                    .conjugate()
                    .transform_vector(&(ecs - Vector3::new(cs_state.x, T::zero(), cs_state.z)));

                Some($fn_ics::<T, Const<{ $params.len() }>, RStride, CStride>(
                    (ecs_norot[0], ecs_norot[1], ecs_norot[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                ))
            }
        }
    };
}

// Implementation of the circular-cylindrical geometry.
impl_xcgm_geom!(
    CCGeometry,
    "Circular-cylindric flux rope geometry.",
    ["phi", "theta", "y", "radius", "x_0"],
    cc_basis,
    cc_detg,
    cc_ecs_to_ics,
    cc_ics_to_ecs
);

// Implementation of the elliptic-cylindrical geometry.
impl_xcgm_geom!(
    ECGeometry,
    "Elliptic-cylindric flux rope geometry.",
    ["phi", "theta", "psi", "y", "delta", "radius", "x_0"],
    ec_basis,
    ec_detg,
    ec_ecs_to_ics,
    ec_ics_to_ecs
);

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_cc_coords() {
        let params =
            SVector::<f64, 5>::from([5.0_f64.to_radians(), -3.0_f64.to_radians(), 0.01, 0.2, 0.0]);

        let mut cs_state = XCState::default();

        CCGeometry::initialize_cs(&params.fixed_rows::<5>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.6, 0.11, 0.5);

        let ecs = CCGeometry::transform_ics_to_ecs(
            &ics_ref.as_view(),
            &params.fixed_rows::<5>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec =
            CCGeometry::transform_ecs_to_ics(&ecs.as_view(), &params.fixed_rows::<5>(0), &cs_state)
                .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        CCGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<5>(0), 1e-6);
    }

    #[test]
    fn test_ec_coords() {
        let params = SVector::<f64, 7>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            5.0_f64.to_radians(),
            0.01,
            1.0,
            0.2,
            0.0,
        ]);

        let mut cs_state = XCState::default();

        ECGeometry::initialize_cs(&params.fixed_rows::<7>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.6, 0.11, 0.5);

        let ecs = ECGeometry::transform_ics_to_ecs(
            &ics_ref.as_view(),
            &params.fixed_rows::<7>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec =
            ECGeometry::transform_ecs_to_ics(&ecs.as_view(), &params.fixed_rows::<7>(0), &cs_state)
                .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        ECGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<7>(0), 1e-6);
    }
}
