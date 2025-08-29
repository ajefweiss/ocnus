//! Implemented models and forward model state types.

mod corem;
mod cylm;
mod wsahux;

pub use corem::*;
pub use cylm::*;
pub use wsahux::*;

/// Concatenates the &str array `b` with an [`OVector<&str>`] `a` and returns an [`ArrayStorage`].
macro_rules! concat_strs {
    ($a: expr, $b: expr) => {{
        const LEN_A: usize = $a.data.0[0].len();

        let mut c = [$b[0]; LEN_A + $b.len()];

        let mut i1 = 0;
        let mut i2 = 0;

        while i1 < LEN_A {
            c[i1] = $a.data.0[0][i1];

            i1 += 1;
        }

        while i2 < $b.len() {
            c[LEN_A + i2] = $b[i2];

            i2 += 1;
        }

        nalgebra::ArrayStorage([c; 1])
    }};
}

// Re-implement the Coordinates trait because we have no inheritance.
// Here we make use of the fact that the parameters for the coords are at the front
// and we pass on smaller fixed views of each parameter vector.
macro_rules! reimpl_coords {
    ($model:ident, $coords: ident, $params: expr) => {
        impl<T, P> Coordinates<T, { $coords::<f32>::PARAMS_COUNT + $params.len() }> for $model<T, P>
        where
            T: Copy + RealField,
        {
            const PARAMS: SVector<&'static str, { $coords::<f32>::PARAMS_COUNT + $params.len() }> =
                SVector::from_array_storage(concat_strs!($coords::<f32>::PARAMS, $params));

            type CSST = XCState<T>;

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<[Vector3<T>; 3]> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn detg<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<T> {
                $coords::detg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut Self::CSST,
            ) {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                $coords::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::PARAMS_COUNT + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                $coords::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f32>::PARAMS_COUNT }>(0),
                    cs_state,
                )
            }
        }
    };
}

pub(crate) use concat_strs;
pub(crate) use reimpl_coords;
