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
            T: nalgebra::RealField;

        impl<T> Default for $model<T>
        where
            T: nalgebra::RealField,
        {
            fn default() -> Self {
                Self(PhantomData::<T>)
            }
        }

        impl<T> bayesfm::geometry::BFMGeometry<T, 3, { $params.len() }> for $model<T>
        where
            T: Copy + Default + nalgebra::RealField,
        {
            const PARAM_NAMES: nalgebra::SVector<&'static str, { $params.len() }> =
                nalgebra::SVector::from_array_storage(nalgebra::ArrayStorage([$params; 1]));

            type CSST = XCState<T>;

            fn contravariant_basis<RStride: nalgebra::Dim, CStride: nalgebra::Dim>(
                ics: &nalgebra::VectorView3<T>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Matrix3<T>> {
                let quaternion = cs_state.q;

                let [dmu, dnu, ds] = $fn_basis::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                );

                Some(nalgebra::Matrix3::from_columns(&[
                    quaternion.transform_vector(&dmu),
                    quaternion.transform_vector(&dnu),
                    quaternion.transform_vector(&ds),
                ]))
            }

            fn sqrt_detg<RStride: nalgebra::Dim, CStride: nalgebra::Dim>(
                ics: &nalgebra::VectorView3<T>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<T> {
                Some($fn_sqrtdetg::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                ))
            }

            fn initialize_csst<RStride: nalgebra::Dim, CStride: nalgebra::Dim>(
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &mut Self::CSST,
            ) {
                let phi = bayesfm::geometry::param_value("phi", &Self::PARAM_NAMES, params);
                let theta = bayesfm::geometry::param_value("theta", &Self::PARAM_NAMES, params);
                let psi = param_value_or_else("psi", &Self::PARAM_NAMES, params, T::zero());
                let radius = bayesfm::geometry::param_value("radius", &Self::PARAM_NAMES, params);
                let x_init = bayesfm::geometry::param_value("x_0", &Self::PARAM_NAMES, params);
                let y = bayesfm::geometry::param_value("y_0", &Self::PARAM_NAMES, params);

                assert!(radius > T::zero(), "radius must be positive");

                cs_state.x = x_init;
                cs_state.z = -radius * y * (T::one() - (phi.sin() * theta.cos()).powi(2)).sqrt()
                    / phi.cos()
                    / theta.cos();

                cs_state.q = quaternion_rot(phi, psi, theta);
            }

            fn transform_internal_to_external<RStride: nalgebra::Dim, CStride: nalgebra::Dim>(
                ics: &nalgebra::VectorView3<T>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Vector3<T>> {
                let quaternion = cs_state.q;

                let ecs_norot = $fn_ecs::<T, { $params.len() }, RStride, CStride>(
                    (ics[0], ics[1], ics[2]),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                );

                Some(
                    quaternion.transform_vector(&ecs_norot)
                        + nalgebra::Vector3::new(cs_state.x, T::zero(), cs_state.z),
                )
            }

            fn transform_external_to_internal<RStride: nalgebra::Dim, CStride: nalgebra::Dim>(
                ecs: &nalgebra::VectorView3<T>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    RStride,
                    CStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Vector3<T>> {
                let quaternion = cs_state.q;

                let ecs_norot = quaternion.conjugate().transform_vector(
                    &(ecs - nalgebra::Vector3::new(cs_state.x, T::zero(), cs_state.z)),
                );

                Some($fn_ics::<T, { $params.len() }, RStride, CStride>(
                    (ecs_norot[0], ecs_norot[1], ecs_norot[2]),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                ))
            }
        }
    };
}

pub(crate) use impl_xcgm_geom;
