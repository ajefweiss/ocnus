/// Concatenates the &str array `b` with an [`nalgebra::OVector<&str>`] `a` and returns an [`nalgebra::ArrayStorage`].
#[macro_export]
macro_rules! model_impl_concat_strs {
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

/// Re-implement the [`crate::coords::Coordinates`]` trait because we have no inheritance.
/// Here we make use of the fact that the parameters for the coordinates are at the front
/// and we pass on smaller fixed views of each parameter vector.
#[macro_export]
macro_rules! model_impl_coords {
    ($model:ident, $csst: ty, $coords: ident, $params: expr) => {
        impl<T, G>
            Coordinates<T, { $coords::<f32>::NDIMS }, { $coords::<f32>::NPARAMS + $params.len() }>
            for $model<T, G>
        where
            T: Copy + RealField,
        {
            const PARAMS: SVector<&'static str, { $coords::<f32>::NPARAMS + $params.len() }> =
                SVector::from_array_storage(model_impl_concat_strs!(
                    $coords::<f32>::PARAMS,
                    $params
                ));

            type CSST = $csst;

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &SVectorView<T, { $coords::<f32>::NDIMS }>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::NPARAMS + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<[SVector<T, { $coords::<f32>::NDIMS }>; { $coords::<f32>::NDIMS }]> {
                $coords::contravariant_basis(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn sqrtdetg<RStride: Dim, CStride: Dim>(
                ics: &SVectorView<T, { $coords::<f32>::NDIMS }>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::NPARAMS + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<T> {
                $coords::sqrtdetg(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::NPARAMS + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut Self::CSST,
            ) {
                $coords::initialize_cs(
                    &params.fixed_rows::<{ $coords::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &SVectorView<T, { $coords::<f32>::NDIMS }>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::NPARAMS + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<SVector<T, { $coords::<f32>::NDIMS }>> {
                $coords::transform_ics_to_ecs::<RStride, CStride>(
                    ics,
                    &params.fixed_rows::<{ $coords::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &SVectorView<T, { $coords::<f32>::NDIMS }>,
                params: &VectorView<
                    T,
                    Const<{ $coords::<f32>::NPARAMS + $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<SVector<T, { $coords::<f32>::NDIMS }>> {
                $coords::transform_ecs_to_ics::<RStride, CStride>(
                    ecs,
                    &params.fixed_rows::<{ $coords::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }
        }
    };
}
