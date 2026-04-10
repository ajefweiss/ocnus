use crate::coords::{Coordinates, param_value};
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

impl<T> Coordinates<T, 3, 4> for SPHGeometry<T>
where
    T: RealField,
{
    const PARAMS: SVector<&'static str, 4> =
        SVector::from_array_storage(ArrayStorage([["x_0", "y_0", "z_0", "radius_0"]; 1]));

    type CSST = SPHState<T>;

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<[Vector3<T>; 3]> {
        let radius = cs_state.radius.clone().clone();

        let r = ics[0].clone();
        let phi = ics[1].clone();
        let theta = ics[2].clone();

        Some([
            Vector3::new(
                phi.clone().cos() * theta.clone().clone().sin(),
                phi.clone().sin() * theta.clone().clone().sin(),
                theta.clone().cos(),
            ) * radius.clone(),
            Vector3::new(
                -phi.clone().sin() * theta.clone().clone().sin(),
                phi.clone().cos() * theta.clone().clone().sin(),
                T::zero(),
            ) * radius.clone()
                * r.clone(),
            Vector3::new(
                phi.clone().cos() * theta.clone().cos(),
                phi.clone().sin() * theta.clone().cos(),
                -theta.clone().clone().sin(),
            ) * radius.clone()
                * r,
        ])
    }

    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<T> {
        let radius = cs_state.radius.clone();

        let r = ics[0].clone();
        let theta = ics[2].clone();

        Some(r.powi(2) * radius.clone().powi(3) * theta.clone().sin())
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &mut Self::CSST,
    ) {
        let x0 = param_value("x_0", &Self::PARAMS, params);
        let y0 = param_value("y_0", &Self::PARAMS, params);
        let z0 = param_value("z_0", &Self::PARAMS, params);
        let radius = param_value("radius_0", &Self::PARAMS, params);

        assert!(radius.clone() > T::zero(), "sphere radius must be positive");

        cs_state.center = Vector3::new(x0, y0, z0);
        cs_state.radius = radius.clone();
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        let center = cs_state.center.clone();
        let radius = cs_state.radius.clone();

        let r = ics[0].clone();
        let phi = ics[1].clone();
        let theta = ics[2].clone();

        Some(
            Vector3::new(
                radius.clone() * r.clone() * phi.clone().cos() * theta.clone().clone().sin(),
                radius.clone() * r.clone() * phi.clone().sin() * theta.clone().clone().sin(),
                radius.clone() * r.clone() * theta.clone().cos(),
            ) + center,
        )
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, U4, RStride, CStride>,
        cs_state: &Self::CSST,
    ) -> Option<Vector3<T>> {
        let center = cs_state.center.clone();
        let radius = cs_state.radius.clone();

        let v = ecs - center;
        let vn = v.norm();

        Some(Vector3::new(
            vn.clone() / radius.clone(),
            v[1].clone().atan2(v[0].clone()),
            (v[2].clone() / vn).acos(),
        ))
    }
}

impl<T> Coordinates<T, 3, 0> for SPHUGeometry<T>
where
    T: RealField,
{
    const PARAMS: SVector<&'static str, 0> = SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<[Vector3<T>; 3]> {
        let r = ics[0].clone();
        let phi = ics[1].clone();
        let theta = ics[2].clone();

        Some([
            Vector3::new(
                phi.clone().cos() * theta.clone().clone().sin(),
                phi.clone().sin() * theta.clone().clone().sin(),
                theta.clone().cos(),
            ),
            Vector3::new(
                -phi.clone().sin() * theta.clone().clone().sin(),
                phi.clone().cos() * theta.clone().clone().sin(),
                T::zero(),
            ) * r.clone(),
            Vector3::new(
                phi.clone().cos() * theta.clone().cos(),
                phi.clone().sin() * theta.clone().cos(),
                -theta.clone().clone().sin(),
            ) * r,
        ])
    }

    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<T> {
        let r = ics[0].clone();
        let theta = ics[2].clone();

        Some(r.powi(2) * theta.clone().clone().sin())
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
        let r = ics[0].clone();
        let phi = ics[1].clone();
        let theta = ics[2].clone();

        Some(Vector3::new(
            r.clone() * phi.clone().cos() * theta.clone().clone().sin(),
            r.clone() * phi.clone().sin() * theta.clone().clone().sin(),
            r * theta.clone().cos(),
        ))
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
        let v = ecs;
        let vn = v.norm();

        Some(Vector3::new(
            vn.clone(),
            v[1].clone().atan2(v[0].clone()),
            (v[2].clone() / vn).acos(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coordinates3D;
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
