use crate::{
    coords::{CoordsError, OcnusCoords},
    fXX,
    math::{T, atan2, cos, powi, quaternion_rot, sin, sqrt},
};
use nalgebra::{Const, Dim, U1, UnitQuaternion, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// coordinate system cs_state type for cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCState<T>
where
    T: fXX,
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
pub fn cc_basis<T, const P: usize, M, CSST, CStride: Dim>(
    (r, nu, _z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let omega = T::two_pi() * nu;
    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    let dr = Vector3::from([cos!(omega), T::zero(), sin!(omega)]) * radius_linearized;
    let dnu =
        Vector3::from([-sin!(omega), T::zero(), cos!(omega)]) * T::two_pi() * r * radius_linearized;
    let dz = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dr, dnu, dz]
}

/// The elliptic-cylindric contravariant basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_basis<T, const P: usize, M, CSST, CStride: Dim>(
    (mu, nu, _s): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    let omega = T::two_pi() * nu;
    let com = cos!(omega);
    let som = sin!(omega);

    let denom = sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

    let nu_nom = T::two_pi() * delta * mu * radius_linearized;

    let dmu = Vector3::from([
        delta * radius_linearized * com / denom,
        T::zero(),
        delta * radius_linearized * som / denom,
    ]);

    let dnu = Vector3::from([
        -nu_nom * powi!(delta, 2) * som / powi!(denom, 3),
        T::zero(),
        nu_nom * com / powi!(denom, 3),
    ]);

    let ds = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dmu, dnu, ds]
}

/// The circular-cylindric metric determinant.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_detg<T, const P: usize, M, CSST, CStride: Dim>(
    (r, _nu, _z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> T
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    T::two_pi() * r * powi!(radius_linearized, 2)
}

/// The elliptic-cylindric metric determinant
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_detg<T, const P: usize, M, CSST, CStride: Dim>(
    (mu, _nu, _s): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> T
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    T::two_pi() * mu * powi!(delta, 2) * powi!(radius_linearized, 2)
}

/// The circular-cylindric coordinate transformation (ecs -> ics).
pub fn cc_ecs_to_ics<T, const P: usize, M, CSST, CStride: Dim>(
    (x, y, z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let radius = M::param_value("radius", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    // Compute polar coordinates (r, omega).
    let r = sqrt!(powi!(x, 2) + powi!(z, 2)) / radius_linearized;

    let mut nu = if r == T::zero() {
        T::zero()
    } else {
        atan2!(z, x) / T::two_pi()
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
pub fn ec_ecs_to_ics<T, const P: usize, M, CSST, CStride: Dim>(
    (x, y, z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    // Compute internal coordinates (mu, nu).
    let r = sqrt!(powi!(x, 2) + powi!(z, 2));

    let (mu, mut nu) = if r == T::zero() {
        (T::zero(), T::zero())
    } else {
        (
            r * sqrt!(powi!(x, 2) + powi!(z, 2) * powi!(delta, 2)) / r / delta / radius_linearized,
            atan2!(z, x) / T!(2.0) / T::pi(),
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
pub fn cc_ics_to_ecs<T, const P: usize, M, CSST, CStride: Dim>(
    (r, nu, y): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let radius = M::param_value("radius", params).unwrap();
    let y_offset = M::param_value("y", params).unwrap();

    let omega = T::two_pi() * nu;
    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    // Compute cartesian coordinates (x, y, z).
    let x = cos!(omega) * r * radius_linearized;
    let z = sin!(omega) * r * radius_linearized;

    Vector3::new(x, y, z)
}

/// The elliptic-cylindric coordinate transformation (ics -> ecs).
pub fn ec_ics_to_ecs<T, const P: usize, M, CSST, CStride: Dim>(
    (mu, nu, s): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusCoords<T, P, CSST>,
    T: fXX,
{
    let y_offset = M::param_value("y", params).unwrap();
    let delta = M::param_value("delta", params).unwrap();
    let radius = M::param_value("radius", params).unwrap();

    let radius_linearized = radius * sqrt!(T::one() - powi!(y_offset, 2));

    let omega = T::two_pi() * nu;

    let df = mu * delta * radius_linearized
        / sqrt!(powi!(cos!(omega), 2) + powi!(delta * sin!(omega), 2));

    // Compute cartesian coordinates (x, y, z).
    let x = cos!(omega) * df;
    let y = sin!(omega) * df;

    Vector3::new(x, s, y)
}

macro_rules! impl_xcgm_geom {
    ($model: ident, $params: expr, $fn_basis: tt, $fn_detg:tt, $fn_ics: tt, $fn_ecs: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        #[derive(Debug)]
        pub struct $model<T>(PhantomData<T>)
        where
            T: fXX;

        impl<T> Default for $model<T>
        where
            T: fXX,
        {
            fn default() -> Self {
                Self(PhantomData::<T>)
            }
        }

        impl<T> OcnusCoords<T, { $params.len() }, XCState<T>> for $model<T>
        where
            T: fXX,
        {
            const PARAMS: [&'static str; { $params.len() }] = $params;

            fn contravariant_basis<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                cs_state: &XCState<T>,
            ) -> Result<[Vector3<T>; 3], CoordsError> {
                let quaternion = cs_state.q;

                let [dmu, dnu, ds] = $fn_basis::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    cs_state,
                );

                Ok([
                    quaternion.transform_vector(&dmu),
                    quaternion.transform_vector(&dnu),
                    quaternion.transform_vector(&ds),
                ])
            }

            /// Compute the determinant of the metric tensor.
            fn detg<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                cs_state: &XCState<T>,
            ) -> Result<T, CoordsError> {
                Ok($fn_detg::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    cs_state,
                ))
            }

            fn initialize_cs<CStride: Dim>(
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                cs_state: &mut XCState<T>,
            ) -> Result<(), CoordsError> {
                let phi = Self::param_value("phi", params).unwrap();
                let theta = Self::param_value("theta", params).unwrap();
                let psi = Self::param_value("psi", params).unwrap_or(T::zero());
                let radius = Self::param_value("radius", params).unwrap();
                let x_init = Self::param_value("x_0", params).unwrap();
                let y = Self::param_value("y", params).unwrap();

                cs_state.x = x_init;
                cs_state.z = radius * y * sqrt!(T::one() - powi!(sin!(phi) * sin!(theta), 2))
                    / cos!(phi)
                    / cos!(theta);

                cs_state.q = quaternion_rot(phi, psi, theta);

                Ok(())
            }

            fn transform_ics_to_ecs<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                cs_state: &XCState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                let quaternion = cs_state.q;

                let ecs_norot = $fn_ecs::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    cs_state,
                );

                Ok(quaternion.transform_vector(&ecs_norot)
                    + Vector3::new(cs_state.x, T::zero(), cs_state.z))
            }

            fn transform_ecs_to_ics<CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                cs_state: &XCState<T>,
            ) -> Result<Vector3<T>, CoordsError> {
                let quaternion = cs_state.q;

                let ecs_norot = quaternion
                    .conjugate()
                    .transform_vector(&(ecs - Vector3::new(cs_state.x, T::zero(), cs_state.z)));

                Ok($fn_ics::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ecs_norot[0], ecs_norot[1], ecs_norot[2]),
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
    ["phi", "theta", "y", "radius", "x_0"],
    cc_basis,
    cc_detg,
    cc_ecs_to_ics,
    cc_ics_to_ecs,
    "Circular-cylindric flux rope geometry."
);

// Implementation of the elliptic-cylindrical geometry.
impl_xcgm_geom!(
    ECGeometry,
    ["phi", "theta", "psi", "y", "delta", "radius", "x_0"],
    ec_basis,
    ec_detg,
    ec_ecs_to_ics,
    ec_ics_to_ecs,
    "Elliptic-cylindric flux rope geometry."
);

#[cfg(test)]
mod tests {
    use crate::coords::{CCGeometry, ECGeometry, OcnusCoords, XCState};
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_cc_coords() {
        let params =
            SVector::<f64, 5>::from([5.0_f64.to_radians(), -3.0_f64.to_radians(), 0.01, 0.2, 0.0]);

        let mut cs_state = XCState::default();

        CCGeometry::initialize_cs(&params.fixed_rows::<5>(0), &mut cs_state).unwrap();

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

        assert!((ics_rec - ics_ref).norm() < 1e-4);

        CCGeometry::test_contravariant_basis(&ics_ref.as_view(), &params.fixed_rows::<5>(0), 1e-5);
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

        ECGeometry::initialize_cs(&params.fixed_rows::<7>(0), &mut cs_state).unwrap();

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

        assert!((ics_rec - ics_ref).norm() < 1e-4);

        ECGeometry::test_contravariant_basis(&ics_ref.as_view(), &params.fixed_rows::<7>(0), 1e-5);
    }
}
