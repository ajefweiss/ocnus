use crate::coords::Coordinates;
use nalgebra::{ArrayStorage, Dim, RealField, SVector, SVectorView, U0, VectorView};
use std::marker::PhantomData;

/// Spherical geometry with arbitrary center position and radius.
pub struct LinearGeometry<T>(PhantomData<T>)
where
    T: RealField;

impl<T> Default for LinearGeometry<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self(PhantomData::<T>)
    }
}

impl<T> Coordinates<T, 1, 0> for LinearGeometry<T>
where
    T: RealField,
{
    const PARAMS: SVector<&'static str, 0> = SVector::from_array_storage(ArrayStorage([[]; 1]));

    type CSST = ();

    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        _ics: &SVectorView<T, 1>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<[SVector<T, 1>; 1]> {
        Some([SVector::from([T::one()])])
    }

    fn sqrtdetg<RStride: Dim, CStride: Dim>(
        _ics: &SVectorView<T, 1>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<T> {
        Some(T::one())
    }

    fn initialize_cs<RStride: Dim, CStride: Dim>(
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &mut Self::CSST,
    ) {
    }

    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &SVectorView<T, 1>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, 1>> {
        Some(ics.clone_owned())
    }

    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &SVectorView<T, 1>,
        _params: &VectorView<T, U0, RStride, CStride>,
        _cs_state: &Self::CSST,
    ) -> Option<SVector<T, 1>> {
        Some(ecs.clone_owned())
    }
}
