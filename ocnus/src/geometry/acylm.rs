use crate::{OcnusModel, OcnusState, math::quaternion_xyz};
use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};

/// Model state for analytical cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCModelState {
    /// Offset along the time axis.
    pub t: f32,

    /// Offset along the x-axis.
    pub x: f32,

    /// Offset along the z-axis.
    pub z: f32,
}

impl OcnusState for XCModelState {}

/// The circular-cylindric coordinate basis vectors.
pub fn cc_basis<const P: usize, M, S, CStride: Dim>(
    (_r, phi, _z): (f32, f32, f32),
    _params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> [Vector3<f32>; 3]
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let dphi = Vector3::from([-phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::zeros(), dphi, dpsi]
}

/// The circular-cylindric coordinate transformation (xyz -> ics)
pub fn cc_ics<const P: usize, M, S, CStride: Dim>(
    (x, y, z): (f32, f32, f32),
    params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute polar coordinates (r, phi).
    let r = (x.powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x);

    Vector3::new(r, phi, y)
}

/// The circular-cylindric coordinate transformation (ics -> xyz)
pub fn cc_xyz<const P: usize, M, S, CStride: Dim>(
    (r, phi, z): (f32, f32, f32),
    params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = phi.cos() * r * radius_linearized;
    let y = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

/// The elliptical-cylindric coordinate basis vectors
pub fn ec_basis<const P: usize, M, S, CStride: Dim>(
    (x, _y, z): (f32, f32, f32),
    params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> [Vector3<f32>; 3]
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let delta = M::get_param_value("delta", params);

    // Compute polar coordinates (r, phi).
    let phi = z.atan2(x / delta);

    let dphi = (x.powi(2) + z.powi(2)).sqrt() * Vector3::from([-delta * phi.sin(), 0.0, phi.cos()]);
    let dpsi = Vector3::from([0.0, 1.0, 0.0]);

    [Vector3::zeros(), dphi, dpsi]
}

/// The elliptical-cylindric coordinate transformation (xyz -> ics)
pub fn ec_ics<const P: usize, M, S, CStride: Dim>(
    (x, y, z): (f32, f32, f32),
    params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let delta = M::get_param_value("delta", params);
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute polar coordinates (r, phi).
    let r = ((x / delta).powi(2) + z.powi(2)).sqrt() / radius_linearized;
    let phi = z.atan2(x / delta);

    Vector3::new(r, phi, y)
}

/// The elliptical-cylindric coordinate transformation (ics -> xyz)
pub fn ec_xyz<const P: usize, M, S, CStride: Dim>(
    (r, phi, z): (f32, f32, f32),
    params: &VectorView<f32, Const<P>, U1, CStride>,
    _state: &XCModelState,
) -> Vector3<f32>
where
    M: OcnusModel<P, S>,
    S: OcnusState,
{
    let delta = M::get_param_value("delta", params);
    let radius = M::get_param_value("radius", params);
    let y_offset = M::get_param_value("y", params);

    let radius_linearized = radius * (1.0 - y_offset.abs()).sqrt();

    // Compute cartesian coordinates (x, y, z).
    let x = phi.cos() * r * radius_linearized * delta;
    let y = phi.sin() * r * radius_linearized;

    Vector3::new(x, y, z)
}

macro_rules! impl_acylm {
    ($model: ident, $params: expr, $param_ranges:expr, $fn_basis: tt, $fn_ics: tt, $fn_xyz: tt, $docs: literal) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        pub struct $model;

        impl OcnusModel<{ $params.len() }, XCModelState> for $model {
            const PARAMS: [&'static str; { $params.len() }] = $params;
            const PARAM_RANGES: [(f32, f32); { $params.len() }] = $param_ranges;

            fn basis_from_ics<CStride: Dim>(
                ics: &VectorView3<f32>,
                vec: &VectorView3<f32>,
                params: &VectorView<f32, Const<{ $params.len() }>, U1, CStride>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let [dr, dphi, dpsi] = Self::coords_basis_ics(ics, params, state);

                let vec = dr * vec[0] + dphi * vec[1] + dpsi * vec[2];

                quaternion.transform_vector(&vec)
            }

            fn coords_basis_ics<CStride: Dim>(
                ics: &VectorView3<f32>,
                params: &VectorView<f32, Const<{ $params.len() }>, U1, CStride>,
                state: &XCModelState,
            ) -> [Vector3<f32>; 3] {
                $fn_basis::<{ $params.len() }, Self, XCModelState, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                )
            }

            fn coords_from_ics<CStride: Dim>(
                ics: &VectorView3<f32>,
                params: &VectorView<f32, Const<{ $params.len() }>, U1, CStride>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);

                let xyz = $fn_xyz::<{ $params.len() }, Self, XCModelState, CStride>(
                    (ics[0], ics[1], ics[2]),
                    params,
                    state,
                );

                quaternion.transform_vector(&xyz) + Vector3::new(state.x, 0.0, state.z)
            }

            fn coords_into_ics<CStride: Dim>(
                xyz: &VectorView3<f32>,
                params: &VectorView<f32, Const<{ $params.len() }>, U1, CStride>,
                state: &XCModelState,
            ) -> Vector3<f32> {
                let phi = Self::get_param_value("phi", params);
                let theta = Self::get_param_value("theta", params);

                let quaternion = quaternion_xyz(phi, 0.0, theta);
                let xyz_t = quaternion
                    .conjugate()
                    .transform_vector(&(xyz - Vector3::new(state.x, 0.0, state.z)));

                $fn_ics::<{ $params.len() }, Self, XCModelState, CStride>(
                    (xyz_t[0], xyz_t[1], xyz_t[2]),
                    params,
                    state,
                )
            }
        }
    };
}

// Implementation of a circular cylindrical flux rope model.
impl_acylm!(
    CCModel,
    ["phi", "theta", "y", "radius"],
    [
        (-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0),
        (0.0, 2.0 * std::f32::consts::PI),
        (-1.0, 1.0),
        (0.01, 0.5),
    ],
    cc_basis,
    cc_ics,
    cc_xyz,
    "Circular-cylindric flux rope model."
);

// Implementation of an elliptical cylindrical flux rope model.
impl_acylm!(
    ECModel,
    ["phi", "theta", "y", "radius", "delta"],
    [
        (-std::f32::consts::PI / 2.0, std::f32::consts::PI / 2.0),
        (0.0, 2.0 * std::f32::consts::PI),
        (-1.0, 1.0),
        (0.01, 0.5),
        (0.05, 1.0),
    ],
    ec_basis,
    ec_ics,
    ec_xyz,
    "Elliptical-cylindric flux rope model."
);
