use crate::{
    coords::OcnusCoords,
    fXX,
    math::{acos, atan2, cos, powi, sin},
};
use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use super::CoordsError;

/// coordinate system cs_state type for a spherical geometry
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct SPHState<T>
where
    T: fXX,
{
    /// Spherical center.
    pub center: Vector3<T>,

    /// Radial scale factor
    pub radius: T,
}

/// Spherical geometry with arbitrary center position and radius.
pub struct SPHGeometry<T>(PhantomData<T>)
where
    T: fXX;

impl<T> Default for SPHGeometry<T>
where
    T: fXX,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> OcnusCoords<T, 4, SPHState<T>> for SPHGeometry<T>
where
    T: fXX,
{
    const PARAMS: [&'static str; 4] = ["center_x0", "center_y0", "center_z0", "radius_0"];

    fn contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        cs_state: &SPHState<T>,
    ) -> Result<[Vector3<T>; 3], CoordsError> {
        let radius = cs_state.radius;

        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Ok([
            Vector3::new(
                cos!(phi) * sin!(theta),
                sin!(phi) * sin!(theta),
                cos!(theta),
            ) * radius,
            Vector3::new(-sin!(phi) * sin!(theta), cos!(phi) * sin!(theta), T::zero()) * radius * r,
            Vector3::new(
                cos!(phi) * cos!(theta),
                sin!(phi) * cos!(theta),
                -sin!(theta),
            ) * radius
                * r,
        ])
    }

    /// Compute the determinant of the metric tensor.
    fn detg<CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        cs_state: &SPHState<T>,
    ) -> Result<T, CoordsError> {
        let radius = cs_state.radius;

        let r = ics[0];
        let theta = ics[2];

        Ok(powi!(r, 2) * powi!(radius, 3) * sin!(theta))
    }

    fn initialize_cs<CStride: Dim>(
        params: &VectorView<T, Const<4>, U1, CStride>,
        cs_state: &mut SPHState<T>,
    ) -> Result<(), CoordsError> {
        let x0 = Self::param_value("center_x0", params).unwrap();
        let y0 = Self::param_value("center_y0", params).unwrap();
        let z0 = Self::param_value("center_z0", params).unwrap();
        let radius = Self::param_value("radius_0", params).unwrap();

        cs_state.center = Vector3::new(x0, y0, z0);
        cs_state.radius = radius;

        Ok(())
    }

    fn transform_ics_to_ecs<CStride: Dim>(
        ics: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        cs_state: &SPHState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let center = cs_state.center;
        let radius = cs_state.radius;

        let r = ics[0];
        let phi = ics[1];
        let theta = ics[2];

        Ok(Vector3::new(
            radius * r * cos!(phi) * sin!(theta),
            radius * r * sin!(phi) * sin!(theta),
            radius * r * cos!(theta),
        ) + center)
    }

    fn transform_ecs_to_ics<CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, Const<4>, U1, CStride>,
        cs_state: &SPHState<T>,
    ) -> Result<Vector3<T>, CoordsError> {
        let center = cs_state.center;
        let radius = cs_state.radius;

        let v = ecs - center;
        let vn = v.norm();

        Ok(Vector3::new(
            vn / radius,
            atan2!(v[1], v[0]),
            acos!(v[2] / vn),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::coords::{OcnusCoords, SPHGeometry, SPHState};
    use nalgebra::{SVector, Vector3};

    #[test]
    fn test_sph_coords() {
        let params = SVector::<f64, 4>::from([0.0, 0.0, 0.0, 0.75]);

        let mut cs_state = SPHState::default();

        SPHGeometry::initialize_cs(&params.fixed_rows::<4>(0), &mut cs_state).unwrap();

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

        assert!((ics_rec - ics_ref).norm() < 1e-4);

        SPHGeometry::test_contravariant_basis(&ics_ref.as_view(), &params.fixed_rows::<4>(0), 1e-5);
    }
}
