//! Implemented atmospheric and geospace Models

mod atmt2o2;
mod atmt5o4;

pub use atmt2o2::*;
pub use atmt5o4::*;

macro_rules! impl_atmos_model {
    ($model: ident, $coeffs: expr, $params: expr) => {
        /// Atmospheric model, assumes isothermal temperature, with respect to height, above the base altitude and only oxygen species.
        #[derive(Clone, Debug, Deserialize, Serialize)]
        pub struct $model<T, G>(G, PhantomData<T>)
        where
            T: Copy + RealField;

        impl<T, G> $model<T, G>
        where
            T: Copy + RealField,
        {
            #[doc = concat!("Create a new [`", stringify!($model), "`].")]
            pub fn new(pdf: G) -> Self {
                Self(pdf, PhantomData::<T>)
            }
        }

        // Re-implement the Coordinates trait because we have no inheritance.
        // Here we make use of the fact that the parameters for the coords are at the front
        // and we pass on smaller fixed views of each parameter vector.
        impl<T, G> Coordinates<T, 3, $params> for $model<T, G>
        where
            T: Copy + RealField,
        {
            const PARAMS: SVector<&'static str, $params> = SVector::from_array_storage(
                model_impl_concat_strs!(CartesianGeometry::<f32>::PARAMS, $coeffs),
            );

            type CSST = ();

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<$params>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<[Vector3<T>; 3]> {
                let v = &params.fixed_rows::<{ CartesianGeometry::<f32>::NPARAMS }>(0);
                CartesianGeometry::contravariant_basis(ics, v, cs_state)
            }

            fn sqrtdetg<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<$params>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<T> {
                CartesianGeometry::sqrtdetg(
                    ics,
                    &params.fixed_rows::<{ CartesianGeometry::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<T, Const<$params>, RStride, CStride>,
                cs_state: &mut Self::CSST,
            ) {
                CartesianGeometry::initialize_cs(
                    &params.fixed_rows::<{ CartesianGeometry::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<$params>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                CartesianGeometry::transform_ics_to_ecs(
                    ics,
                    &params.fixed_rows::<{ CartesianGeometry::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<T, Const<$params>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                CartesianGeometry::transform_ecs_to_ics(
                    ecs,
                    &params.fixed_rows::<{ CartesianGeometry::<f32>::NPARAMS }>(0),
                    cs_state,
                )
            }
        }

        impl<T, G> Model<T, 3, $params> for $model<T, G>
        where
            T: AsPrimitive<usize> + Copy + Default + RealField,
            G:  Density<T, Const<$params>>,
            for<'a> &'a G: Density<T, Const<$params>>,
        {
            const RCS: usize = $params;

            type FMST = ();

            fn domain(&self) -> impl Domain<T, Const<$params>>  {
                self.0.domain().clone()
            }

            fn evolve_state(
                &self,
                _time_step: T,
                _params: &VectorView<T, Const<$params>, U1, Const<$params>>,
                _fm_state: &mut Self::FMST,
                _cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Ok(())
            }

            fn initialize_states(
                &self,
                params: &VectorView<T, Const<$params>>,
                _fm_state: &mut Self::FMST,
                cs_state: &mut Self::CSST,
            ) -> Result<(), ModelError<T>> {
                Self::initialize_cs(params, cs_state);

                Ok(())
            }

            fn prior(&self) -> impl Density<T, Const<$params>>  {
                self.0.clone()
            }
        }
    };
}

macro_rules! spherical_harmonics {
    ($offset: expr, $lmax: expr, $coeffs: expr, $theta: expr, $varphi: expr) => {
        (0..($lmax + 1))
            .map(|ldx| {
                let mut sum = T::zero();

                (-ldx..(ldx + 1)).for_each(|mdx| {
                    sum += $coeffs[$offset + (ldx * (ldx + 1) + mdx) as usize]
                        * sph_yml(ldx as usize, mdx)($theta, $varphi);
                });

                sum
            })
            .sum::<T>()
    };
}

pub(crate) use impl_atmos_model;
pub(crate) use spherical_harmonics;
