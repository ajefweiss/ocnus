use crate::{geom::OcnusGeometry, math::quaternion_xyz};
use nalgebra::{Const, Dim, RealField, SimdRealField, U1, Vector3, VectorView, VectorView3};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Model state for analytical cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCState<T> {
    /// Offset along the time axis.
    pub t: T,

    /// Offset along the x-axis.
    pub x: T,

    /// Offset along the z-axis.
    pub z: T,
}

/// The circular-cylindric coordinate basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn cc_basis<T, const P: usize, M, GS, CStride: Dim>(
    (_r, phi, _z): (T, T, T),
    _params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    M: OcnusGeometry<T, P, GS>,
    T: 'static + Debug + Float,
{
    let dr = Vector3::from([phi.cos(), T::zero(), phi.sin()]);
    let dphi = Vector3::from([-phi.sin(), T::zero(), phi.cos()]);
    let dpsi = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dr, dphi, dpsi]
}

/// The (non-orthogonal) elliptic-cylindric coordinate basis vectors.
#[allow(clippy::extra_unused_type_parameters)]
pub fn ec_basis<T, const P: usize, M, GS, CStride: Dim>(
    (mu, nu, _s): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> [Vector3<T>; 3]
where
    M: OcnusGeometry<T, P, GS>,
    T: 'static + Debug + Float + RealField,
{
    let psi = M::param_value("psi", params);
    let y_offset = M::param_value("y", params);
    let delta = M::param_value("delta", params);
    let radius = M::param_value("radius", params);

    let radius_linearized = radius * Float::sqrt(T::one() - Float::powi(y_offset, 2));

    let omega = T::from_f64(2.0).unwrap() * T::pi() * (nu - psi);
    let com = Float::cos(omega);
    let som = Float::sin(omega);

    let denom =
        Float::sqrt(Float::powi(Float::cos(omega), 2) + Float::powi(delta * Float::sin(omega), 2));

    let phi_nom = T::from_f64(2.0).unwrap() * T::pi() * delta * mu * radius_linearized;

    let dmu = Vector3::from([
        delta * radius_linearized * com / denom,
        T::zero(),
        delta * radius_linearized * som / denom,
    ]);
    let dnu = Vector3::from([
        -phi_nom * Float::powi(delta, 2) * som / Float::powi(denom, 3),
        T::zero(),
        phi_nom * com / Float::powi(denom, 3),
    ]);
    let ds = Vector3::from([T::zero(), T::one(), T::zero()]);

    [dmu, dnu, ds]
}

/// The circular-cylindric coordinate transformation (xyz -> ics).
pub fn cc_xyz_to_ics<T, const P: usize, M, GS, CStride: Dim>(
    (x, y, z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusGeometry<T, P, GS>,
    T: Float,
{
    let radius = M::param_value("radius", params);
    let y_offset = M::param_value("y", params);

    let radius_linearized = radius * (T::one() - y_offset.abs().powi(2)).sqrt();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x);

    Vector3::new(r, phi, y)
}

/// The elliptic-cylindric coordinate transformation (xyz -> ics).
pub fn ec_xyz_to_ics<T, const P: usize, M, GS, CStride: Dim>(
    (x, y, z): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusGeometry<T, P, GS>,
    T: Float + FromPrimitive,
{
    let psi = M::param_value("psi", params);
    let y_offset = M::param_value("y", params);
    let delta = M::param_value("delta", params);
    let radius = M::param_value("radius", params);

    let radius_linearized = radius * Float::sqrt(T::one() - Float::powi(y_offset, 2));

    // Compute intrinsic coordinates (mu, nu).
    let r = (x.powi(2) + z.powi(2)).sqrt();
    let mu = r * (x.powi(2) + z.powi(2) * delta.powi(2)).sqrt() / r / delta / radius_linearized;
    let nu = psi - (x / r).acos() / T::from_f64(2.0 * std::f64::consts::PI).unwrap();

    Vector3::new(mu, nu, y)
}

/// The circular-cylindric coordinate transformation (ics -> xyz).
pub fn cc_ics_to_xyz<T, const P: usize, M, GS, CStride: Dim>(
    (r, phi, y): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusGeometry<T, P, GS>,
    T: Float,
{
    let radius = M::param_value("radius", params);
    let y_offset = M::param_value("y", params);

    let radius_linearized = radius * (T::one() - y_offset.abs().powi(2)).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = phi.cos() * r * radius_linearized;
    let z = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

/// The elliptic-cylindric coordinate transformation (ics -> xyz).
pub fn ec_ics_to_xyz<T, const P: usize, M, GS, CStride: Dim>(
    (mu, nu, s): (T, T, T),
    params: &VectorView<T, Const<P>, U1, CStride>,
    _state: &XCState<T>,
) -> Vector3<T>
where
    M: OcnusGeometry<T, P, GS>,
    T: Float + FromPrimitive + RealField,
{
    let psi = M::param_value("psi", params);
    let y_offset = M::param_value("y", params);
    let delta = M::param_value("delta", params);
    let radius = M::param_value("radius", params);

    let radius_linearized = radius * Float::sqrt(T::one() - Float::powi(y_offset, 2));

    let omega = T::from_f64(2.0).unwrap() * T::pi() * (nu - psi);

    let df = mu * delta * radius_linearized
        / Float::sqrt(
            Float::powi(Float::cos(omega), 2) + Float::powi(delta * Float::sin(omega), 2),
        );

    // Compute cartesian coordinates (x, y, z).
    let x = Float::cos(omega) * df;
    let y = Float::sin(omega) * df;

    Vector3::new(x, s, y)
}

macro_rules! impl_xcgm_geom {
    ($model: ident, $params: expr, $fn_basis: tt, $fn_ics: tt, $fn_xyz: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        #[derive(Debug)]
        pub struct $model<T>(PhantomData<T>)
        where
            T: Float + RealField + SimdRealField;

        impl<T> Default for $model<T>
        where
            T: Float + RealField + SimdRealField,
        {
            fn default() -> Self {
                Self(PhantomData::<T>)
            }
        }

        impl<T> OcnusGeometry<T, { $params.len() }, XCState<T>> for $model<T>
        where
            T: Float + RealField + SimdRealField,
        {
            const PARAMS: [&'static str; { $params.len() }] = $params;

            fn basis_vectors<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &XCState<T>,
            ) -> [Vector3<T>; 3] {
                $fn_basis::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    geom_state,
                )
            }

            fn coords_ics_to_xyz<CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &XCState<T>,
            ) -> Vector3<T> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, T::zero(), theta);

                let xyz = $fn_xyz::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    geom_state,
                );

                quaternion.transform_vector(&xyz)
                    + Vector3::new(geom_state.x, T::zero(), geom_state.z)
            }

            fn coords_xyz_to_ics<CStride: Dim>(
                xyz: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &XCState<T>,
            ) -> Vector3<T> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, T::zero(), theta);
                let xyz_t = quaternion
                    .conjugate()
                    .transform_vector(&(xyz - Vector3::new(geom_state.x, T::zero(), geom_state.z)));

                $fn_ics::<T, { $params.len() }, Self, XCState<T>, CStride>(
                    (xyz_t[0], xyz_t[1], xyz_t[2]),
                    params,
                    geom_state,
                )
            }

            fn create_xyz_vector<CStride: Dim>(
                ics: &VectorView3<T>,
                vec: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &XCState<T>,
            ) -> Vector3<T> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, T::zero(), theta);

                let [d1, d2, d3] = Self::basis_vectors(ics, params, geom_state);

                let vec = d1 * vec[0] + d2 * vec[1] + d3 * vec[2];

                quaternion.transform_vector(&vec)
            }

            fn geom_state<CStride: Dim>(
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &mut XCState<T>,
            ) {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);
                let radius = Self::param_value("radius", params);
                let x_init = Self::param_value("x_0", params);
                let y = Self::param_value("y", params);

                geom_state.t = T::zero();
                geom_state.x = x_init;
                geom_state.z = radius
                    * y
                    * Float::sqrt(T::one() - Float::powi(Float::sin(phi) * Float::sin(theta), 2))
                    / Float::cos(phi)
                    / Float::cos(theta);
            }
        }
    };
}

// Implementation of a circular-cylindrical geometry.
impl_xcgm_geom!(
    CCModel,
    ["phi", "theta", "y", "radius", "x_0"],
    cc_basis,
    cc_xyz_to_ics,
    cc_ics_to_xyz,
    "Circular-cylindric flux rope geometry."
);

// Implementation of a elliptic-cylindrical geometry.
impl_xcgm_geom!(
    ECModel,
    ["phi", "theta", "psi", "y", "delta", "radius", "x_0"],
    ec_basis,
    ec_xyz_to_ics,
    ec_ics_to_xyz,
    "Elliptic-cylindric flux rope geometry."
);

#[cfg(test)]
mod tests {
    use nalgebra::SVector;

    use super::{ECModel, XCState, ec_ics_to_xyz, ec_xyz_to_ics};

    #[test]
    fn test_ec_coords() {
        let params = SVector::<f64, 7>::from([
            5.0_f64.to_radians(),
            -3.0_f64.to_radians(),
            0.5,
            0.01,
            1.0,
            0.2,
            0.0,
        ]);

        let state = XCState::default();

        let xyz = ec_ics_to_xyz::<_, 7, ECModel<f64>, _, _>(
            (0.6, 0.11, 0.5),
            &params.fixed_rows::<7>(0),
            &state,
        );

        println!("{}", xyz);

        let ics = ec_xyz_to_ics::<_, 7, ECModel<f64>, _, _>(
            (xyz[0], xyz[1], xyz[2]),
            &params.fixed_rows::<7>(0),
            &state,
        );

        println!("{}", ics);
    }
}
