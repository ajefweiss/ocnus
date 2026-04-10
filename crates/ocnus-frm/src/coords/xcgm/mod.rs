mod cc;
mod ec;

pub use cc::*;
pub use ec::*;

use nalgebra::{RealField, UnitQuaternion};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Coordinate system state type for cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct XCState<T>
where
    T: RealField,
{
    /// Offset along the x-axis.
    pub x: T,

    /// Offset along the z-axis.
    pub z: T,

    /// Quaternion for orientation.
    pub q: UnitQuaternion<T>,
}

macro_rules! impl_xcgm_geom {
    ($model: ident, $docs: literal, $params: expr, $fn_basis: tt, $fn_sqrtdetg:tt, $fn_ics: tt, $fn_ecs: tt) => {
        #[doc=$docs]
        #[allow(non_camel_case_types)]
        #[derive(Debug)]
        pub struct $model<T>(PhantomData<T>)
        where
            T: RealField;

        impl<T> Default for $model<T>
        where
            T: RealField,
        {
            fn default() -> Self {
                Self(PhantomData::<T>)
            }
        }

        impl<T> Coordinates<T, 3, { $params.len() }> for $model<T>
        where
            T: Copy + RealField,
        {
            const PARAMS: SVector<&'static str, { $params.len() }> =
                SVector::from_array_storage(ArrayStorage([$params; 1]));

            type CSST = XCState<T>;

            fn contravariant_basis<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<[Vector3<T>; 3]> {
                let quaternion = cs_state.q;

                let [dmu, dnu, ds] = $fn_basis::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                );

                Some([
                    quaternion.transform_vector(&dmu),
                    quaternion.transform_vector(&dnu),
                    quaternion.transform_vector(&ds),
                ])
            }

            fn sqrtdetg<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<T> {
                Some($fn_sqrtdetg::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                ))
            }

            fn initialize_cs<RStride: Dim, CStride: Dim>(
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &mut Self::CSST,
            ) {
                let phi = param_value("phi", &Self::PARAMS, params);
                let theta = param_value("theta", &Self::PARAMS, params);
                let psi = param_value_or_else("psi", &Self::PARAMS, params, T::zero());
                let radius = param_value("radius", &Self::PARAMS, params);
                let x_init = param_value("x_0", &Self::PARAMS, params);
                let y = param_value("y_0", &Self::PARAMS, params);

                assert!(radius > T::zero(), "radius must be positive");

                cs_state.x = x_init;
                cs_state.z = -radius * y * (T::one() - (phi.sin() * theta.cos()).powi(2)).sqrt()
                    / phi.cos()
                    / theta.cos();

                cs_state.q = quaternion_rot(phi, psi, theta);
            }

            fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
                ics: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                let quaternion = cs_state.q;

                let ecs_norot = $fn_ecs::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                );

                Some(
                    quaternion.transform_vector(&ecs_norot)
                        + Vector3::new(cs_state.x, T::zero(), cs_state.z),
                )
            }

            fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
                ecs: &VectorView3<T>,
                params: &VectorView<T, Const<{ $params.len() }>, RStride, CStride>,
                cs_state: &Self::CSST,
            ) -> Option<Vector3<T>> {
                let quaternion = cs_state.q;

                let ecs_norot = quaternion
                    .conjugate()
                    .transform_vector(&(ecs - Vector3::new(cs_state.x, T::zero(), cs_state.z)));

                Some($fn_ics::<T, { $params.len() }, RStride, CStride>(
                    (ecs_norot[0], ecs_norot[1], ecs_norot[2]),
                    &Self::PARAMS,
                    params,
                    cs_state,
                ))
            }
        }
    };
}

pub(crate) use impl_xcgm_geom;
