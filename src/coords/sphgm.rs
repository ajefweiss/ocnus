use crate::coords::{OcnusCoords, param_value};
use nalgebra::{ArrayStorage, Dim, RealField, SVector, U0, U4, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

/// Coordinate system state type for a spherical geometry
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SPHState<T>
where
    T: RealField,
{
    /// Spherical center.
    pub center: Vector3<T>,

    /// Radial scale factor
    pub radius: T,
}

/// Spherical geometry with arbitrary center position and radius.
pub struct SPHGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for SPHGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

/// Spherical geometry for a unit sphere centered on the origin.
pub struct SPHUGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for SPHUGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusCoords<T, 4, SPHState<T>> for SPHGeometry<T>
where
    T: Copy + RealField,
{
    const PARAMS: SVector<&'static str, 4> = SVector::from_array_storage(ArrayStorage(
        [["center_x0", "center_y0", "center_z0", "radius_0"]; 1],
    ));

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &SPHState<T>,
    ) -> Option<[Vector3<T>; 3]> {
        let radius = cs_state.radius;

        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Some([
            Vector3::new(
                phi.cos() * theta.sin(),
                phi.sin() * theta.sin(),
                theta.cos(),
            ) * radius,
            Vector3::new(-phi.sin() * theta.sin(), phi.cos() * theta.sin(), T::zero()) * radius * r,
            Vector3::new(
                phi.cos() * theta.cos(),
                phi.sin() * theta.cos(),
                -theta.sin(),
            ) * radius
                * r,
        ])
    }

    /// Compute the determinant of the metric tensor.
    fn detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &SPHState<T>,
    ) -> Option<T> {
        let radius = cs_state.radius;

        let r = ics[0];
        let theta = ics[2];

        Some(r.powi(2) * radius.powi(3) * theta.sin())
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &mut SPHState<T>,
    ) {
        let x0 = param_value("center_x0", &Self::PARAMS, params).unwrap();
        let y0 = param_value("center_y0", &Self::PARAMS, params).unwrap();
        let z0 = param_value("center_z0", &Self::PARAMS, params).unwrap();
        let radius = param_value("radius_0", &Self::PARAMS, params).unwrap();

        assert!(radius > T::zero(), "sphere radius must be positive");

        cs_state.center = Vector3::new(x0, y0, z0);
        cs_state.radius = radius;
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &SPHState<T>,
    ) -> Option<Vector3<T>> {
        let center = cs_state.center;
        let radius = cs_state.radius;

        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Some(
            Vector3::new(
                radius * r * phi.cos() * theta.sin(),
                radius * r * phi.sin() * theta.sin(),
                radius * r * theta.cos(),
            ) + center,
        )
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &SPHState<T>,
    ) -> Option<Vector3<T>> {
        let center = cs_state.center;
        let radius = cs_state.radius;

        let v = ecs - center;
        let vn = v.norm();

        Some(Vector3::new(
            vn / radius,
            v[1].atan2(v[0]),
            (v[2] / vn).acos(),
        ))
    }
}

impl<T> OcnusCoords<T, 0, ()> for SPHUGeometry<T>
where
    T: Copy + RealField,
{
    const PARAMS: SVector<&'static str, 0> = SVector::from_array_storage(ArrayStorage([[]; 1]));

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<[Vector3<T>; 3]> {
        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Some([
            Vector3::new(
                phi.cos() * theta.sin(),
                phi.sin() * theta.sin(),
                theta.cos(),
            ),
            Vector3::new(-phi.sin() * theta.sin(), phi.cos() * theta.sin(), T::zero()) * r,
            Vector3::new(
                phi.cos() * theta.cos(),
                phi.sin() * theta.cos(),
                -theta.sin(),
            ) * r,
        ])
    }

    /// Compute the determinant of the metric tensor.
    fn detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<T> {
        let r = ics[0];
        let theta = ics[2];

        Some(r.powi(2) * theta.sin())
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &mut (),
    ) {
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Some(Vector3::new(
            r * phi.cos() * theta.sin(),
            r * phi.sin() * theta.sin(),
            r * theta.cos(),
        ))
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
        let v = ecs;
        let vn = v.norm();

        Some(Vector3::new(vn, v[1].atan2(v[0]), (v[2] / vn).acos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_sph_coords() {
        let params = SVector::<f64, 4>::from([0.0, 0.0, 0.0, 0.75]);

        let mut cs_state = SPHState::default();

        SPHGeometry::initialize_cs(&params.fixed_rows::<4>(0), &mut cs_state);

        let ics_ref = Vector3::new(0.56, 0.17, 0.45);

        let ecs = SPHGeometry::transform_ics_to_ecs(
            &ics_ref.as_view(),
            &params.fixed_rows::<4>(0),
            &cs_state,
        )
        .unwrap();

        let ics_rec = SPHGeometry::transform_ecs_to_ics(
            &ecs.as_view(),
            &params.fixed_rows::<4>(0),
            &cs_state,
        )
        .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        SPHGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<4>(0), 1e-6);
    }

    #[test]
    fn test_cnsph_coords() {
        let params = SVector::<f64, 0>::from([]);

        SPHUGeometry::initialize_cs(&params.fixed_rows::<0>(0), &mut ());

        let ics_ref = Vector3::new(0.56, 0.17, 0.45);

        let ecs =
            SPHUGeometry::transform_ics_to_ecs(&ics_ref.as_view(), &params.fixed_rows::<0>(0), &())
                .unwrap();

        let ics_rec =
            SPHUGeometry::transform_ecs_to_ics(&ecs.as_view(), &params.fixed_rows::<0>(0), &())
                .unwrap();

        assert!((ics_rec - ics_ref).norm() < 1e-6);

        SPHUGeometry::test_implementation(&ics_ref.as_view(), &params.fixed_rows::<0>(0), 1e-6);
    }
}
