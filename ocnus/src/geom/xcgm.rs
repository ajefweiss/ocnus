use crate::{geom::OcnusGeometry, math::quaternion_xyz};
use nalgebra::{Const, Dim, Scalar, SimdRealField, U1, Vector3, VectorView, VectorView3};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

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
    let dphi = Vector3::from([-phi.sin(), T::zero(), phi.cos()]);
    let dpsi = Vector3::from([T::zero(), T::one(), T::zero()]);

    [Vector3::zeros(), dphi, dpsi]
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

/// The circular-cylindric coordinate transformation (ics -> xyz).
pub fn cc_ics_to_xyz<T, const P: usize, M, GS, CStride: Dim>(
    (r, phi, z): (T, T, T),
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
    let y = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

macro_rules! impl_acylm {
    ($model: ident, $params: expr, $fn_basis: tt, $fn_ics: tt, $fn_xyz: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        pub struct $model;

        impl<T> OcnusGeometry<T, { $params.len() }, XCState<T>> for $model
        where
            T: 'static + Float + Scalar + SimdRealField + TryFrom<f64>,
            <T as nalgebra::SimdValue>::Element: nalgebra::RealField,
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

            fn create_xyz_vector<CStride: Dim>(
                ics: &VectorView3<T>,
                vec: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, U1, CStride>,
                geom_state: &XCState<T>,
            ) -> Vector3<T> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, T::zero(), theta);

                let [dr, dphi, dpsi] = Self::basis_vectors(ics, params, geom_state);

                let vec = dr * vec[0] + dphi * vec[1] + dpsi * vec[2];

                quaternion.transform_vector(&vec)
            }

            fn initialize_geom_state<CStride: Dim>(
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
                geom_state.z =
                    radius * y * ((T::one() - (phi.sin() * theta.cos()).powi(2)) as T).sqrt()
                        / phi.cos()
                        / theta.cos();
            }
        }
    };
}

// Implementation of a circular cylindrical flux rope model.
impl_acylm!(
    CCModel,
    ["phi", "theta", "y", "radius"],
    cc_basis,
    cc_xyz_to_ics,
    cc_ics_to_xyz,
    "Circular-cylindric flux rope model."
);
