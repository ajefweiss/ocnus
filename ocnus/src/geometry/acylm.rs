use crate::{geometry::OcnusGeometry, math::quaternion_xyz};
use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};

/// Model state for analytical cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCState {
    /// Offset along the time axis.
    pub t: f64,

    /// Offset along the x-axis.
    pub x: f64,

    /// Offset along the z-axis.
    pub z: f64,
}

/// The circular-cylindric coordinate basis vectors.
pub fn cc_basis<const P: usize, M, GS, CStride: Dim>(
    (_r, phi, _z): (f64, f64, f64),
    _params: &VectorView<f64, Const<P>, U1, CStride>,
    _state: &XCState,
) -> [Vector3<f64>; 3]
where
    M: OcnusGeometry<P, GS>,
{
    let dphi = Vector3::from([-phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::zeros(), dphi, dpsi]
}

/// The circular-cylindric coordinate transformation (xyz -> ics)
pub fn cc_ics<const P: usize, M, GS, CStride: Dim>(
    (x, y, z): (f64, f64, f64),
    params: &VectorView<f64, Const<P>, U1, CStride>,
    _state: &XCState,
) -> Vector3<f64>
where
    M: OcnusGeometry<P, GS>,
{
    let radius = M::param_value("radius", params);
    let y_offset = M::param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x);

    Vector3::new(r, phi, y)
}

/// The circular-cylindric coordinate transformation (ics -> xyz)
pub fn cc_xyz<const P: usize, M, GS, CStride: Dim>(
    (r, phi, z): (f64, f64, f64),
    params: &VectorView<f64, Const<P>, U1, CStride>,
    _state: &XCState,
) -> Vector3<f64>
where
    M: OcnusGeometry<P, GS>,
{
    let radius = M::param_value("radius", params);
    let y_offset = M::param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = phi.cos() * r * radius_linearized;
    let y = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

macro_rules! impl_acylm {
    ($model: ident, $params: expr, $param_ranges:expr, $fn_basis: tt, $fn_ics: tt, $fn_xyz: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        pub struct $model;

        impl OcnusGeometry<{ $params.len() }, XCState> for $model {
            const PARAMS: [&'static str; { $params.len() }] = $params;
            const PARAM_RANGES: [(f64, f64); { $params.len() }] = $param_ranges;

            fn coords_basis<CStride: Dim>(
                ics: &VectorView3<f64>,
                params: &VectorView<f64, Const<{ $params.len() }>, U1, CStride>,
                state: &XCState,
            ) -> [Vector3<f64>; 3] {
                $fn_basis::<{ $params.len() }, Self, XCState, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                )
            }

            fn coords_ics<CStride: Dim>(
                xyz: &VectorView3<f64>,
                params: &VectorView<f64, Const<{ $params.len() }>, U1, CStride>,
                state: &XCState,
            ) -> Vector3<f64> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);
                let xyz_t = quaternion
                    .conjugate()
                    .transform_vector(&(xyz - Vector3::new(state.x, 0.0, state.z)));

                $fn_ics::<{ $params.len() }, Self, XCState, CStride>(
                    (xyz_t[0], xyz_t[1], xyz_t[2]),
                    params,
                    state,
                )
            }

            fn coords_xyz<CStride: Dim>(
                ics: &VectorView3<f64>,
                params: &VectorView<f64, Const<{ $params.len() }>, U1, CStride>,
                state: &XCState,
            ) -> Vector3<f64> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let xyz = $fn_xyz::<{ $params.len() }, Self, XCState, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                );

                quaternion.transform_vector(&xyz) + Vector3::new(state.x, 0.0, state.z)
            }

            fn coords_xyz_vector<CStride: Dim>(
                ics: &VectorView3<f64>,
                vec: &VectorView3<f64>,
                params: &VectorView<f64, Const<{ $params.len() }>, U1, CStride>,
                state: &XCState,
            ) -> Vector3<f64> {
                let phi = Self::param_value("phi", params);
                let theta = Self::param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let [dr, dphi, dpsi] = Self::coords_basis(ics, params, state);

                let vec = dr * vec[0] + dphi * vec[1] + dpsi * vec[2];

                quaternion.transform_vector(&vec)
            }
        }
    };
}

// Implementation of a circular cylindrical flux rope model.
impl_acylm!(
    CCModel,
    ["phi", "theta", "y", "radius"],
    [
        (-std::f64::consts::PI / 2.0, std::f64::consts::PI / 2.0),
        (0.0, 2.0 * std::f64::consts::PI),
        (-1.0, 1.0),
        (0.01, 0.5),
    ],
    cc_basis,
    cc_ics,
    cc_xyz,
    "Circular-cylindric flux rope model."
);
