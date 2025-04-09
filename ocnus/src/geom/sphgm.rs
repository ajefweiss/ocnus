use crate::geom::OcnusGeometry;
use nalgebra::{
    Const, Dim, RealField, Scalar, SimdRealField, U1, Vector3, VectorView, VectorView3,
};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Model state for a spherical geometry
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SPHState<T>
where
    T: Clone + Scalar,
{
    /// Spherical center.
    pub center: Vector3<T>,

    /// Radial scale factor
    pub radius: T,
}

/// Spherical geometry with arbitrary center position and radius.
pub struct SPHGeometry<T>(PhantomData<T>)
where
    T: Float + RealField + SimdRealField;

impl<T> Default for SPHGeometry<T>
where
    T: Float + RealField + SimdRealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusGeometry<T, 4, SPHState<T>> for SPHGeometry<T>
where
    T: Float + RealField + SimdRealField,
{
    const PARAMS: [&'static str; 4] = ["center_x0", "center_y0", "center_z0", "radius"];

    fn basis_vectors<CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        _geom_state: &SPHState<T>,
    ) -> [Vector3<T>; 3] {
        let phi = ics[1];
        let theta = ics[2];

        [
            Vector3::new(
                Float::sin(theta) * Float::cos(phi),
                Float::sin(theta) * Float::sin(phi),
                Float::cos(theta),
            ),
            Vector3::new(-Float::sin(phi), Float::cos(phi), T::zero()),
            Vector3::new(
                Float::cos(theta) * Float::cos(phi),
                Float::cos(theta) * Float::sin(phi),
                Float::cos(theta),
            ),
        ]
    }

    fn coords_ics_to_xyz<CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        geom_state: &SPHState<T>,
    ) -> Vector3<T> {
        let center = geom_state.center;
        let radius = geom_state.radius;

        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Vector3::new(
            radius * r * Float::cos(phi) * Float::sin(theta),
            radius * r * Float::sin(phi) * Float::sin(theta),
            radius * r * Float::cos(theta),
        ) + center
    }

    fn coords_xyz_to_ics<CStride: Dim>(
        xyz: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        geom_state: &SPHState<T>,
    ) -> Vector3<T> {
        let center = geom_state.center;
        let radius = geom_state.radius;

        let v = xyz - center;
        let vn = v.norm();

        Vector3::new(
            vn / radius,
            Float::acos(v[2] / vn),
            Float::atan2(v[1], v[0]),
        )
    }

    fn create_xyz_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        vec: &VectorView3<T>,
        params: &VectorView<T, Const<4>, U1, CStride>,
        geom_state: &SPHState<T>,
    ) -> Vector3<T> {
        let [d1, d2, d3] = Self::basis_vectors(ics, params, geom_state);

        d1 * vec[0] + d2 * vec[1] + d3 * vec[2]
    }

    fn geom_state<CStride: Dim>(
        params: &VectorView<T, Const<4>, U1, CStride>,
        geom_state: &mut SPHState<T>,
    ) {
        let x0 = Self::param_value("center_x0", params);
        let y0 = Self::param_value("center_y0", params);
        let z0 = Self::param_value("center_z0", params);
        let radius = Self::param_value("radius", params);

        geom_state.center = Vector3::new(x0, y0, z0);
        geom_state.radius = radius;
    }
}
