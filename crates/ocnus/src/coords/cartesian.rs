use crate::coords::Coordinates;
use nalgebra::{ArrayStorage, Dim, RealField, SVector, U0, Vector3, VectorView, VectorView3};
use std::marker::PhantomData;

/// Spherical geometry for a unit sphere centered on the origin.
pub struct CartesianGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for CartesianGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> Coordinates<T, 3, 0> for CartesianGeometry<T>
where
    T: RealField,
{
    const PARAMS: SVector<&'static str, 0> = SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        _ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<[Vector3<T>; 3]> {
        Some([
            Vector3::x_axis().into_inner(),
            Vector3::y_axis().into_inner(),
            Vector3::z_axis().into_inner(),
        ])
    }

    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        _ics: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<T> {
        Some(T::one())
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
        Some(ics.clone_owned())
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &(),
    ) -> Option<Vector3<T>> {
        Some(ecs.clone_owned())
    }
}
