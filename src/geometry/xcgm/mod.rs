mod cc;
mod ec;

pub use cc::*;
pub use ec::*;

use nalgebra::{RealField, UnitQuaternion};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Coordinate system state type for cylindrical models with arbitrary cross-section shapes.
#[derive(Clone, Debug, Deserialize, Serialize)]
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

impl<T> Default for XCState<T>
where
    T: RealField,
{
    fn default() -> Self {
        Self {
            x: T::zero(),
            z: T::zero(),
            q: UnitQuaternion::identity(),
        }
    }
}

macro_rules! impl_xcgm_geometry {
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

        impl<T> bayesfm::geometry::Geometry<T, 3, { $params.len() }> for $model<T>
        where
            T: nalgebra::RealField,
        {
            const PARAM_NAMES: nalgebra::SVector<&'static str, { $params.len() }> =
                nalgebra::SVector::from_array_storage(nalgebra::ArrayStorage([$params; 1]));

            type CSST = XCState<T>;

            fn contravariant_basis<CRStride, CCStride, PRStride, PCStride>(
                ics: &nalgebra::VectorView<T, nalgebra::Const<3>, CRStride, CCStride>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Matrix3<T>>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                let quaternion = cs_state.q.clone();

                let [dmu, dnu, ds] = $fn_basis::<T, { $params.len() }, PRStride, PCStride>(
                    (ics[0].clone(), ics[1].clone(), ics[2].clone()),
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

            fn sqrt_detg<CRStride, CCStride, PRStride, PCStride>(
                ics: &nalgebra::VectorView<T, nalgebra::Const<3>, CRStride, CCStride>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<T>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                Some($fn_sqrtdetg::<T, { $params.len() }, PRStride, PCStride>(
                    (ics[0].clone(), ics[1].clone(), ics[2].clone()),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                ))
            }

            fn initialize_csst<PRStride: nalgebra::Dim, PCStride: nalgebra::Dim>(
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    PRStride,
                    PCStride,
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
                cs_state.z = -radius
                    * y
                    * (T::one() - (phi.clone().sin() * theta.clone().cos()).powi(2)).sqrt()
                    / phi.clone().cos()
                    / theta.clone().cos();

                cs_state.q = quaternion_rot(phi, psi, theta);
            }

            fn transform_internal_to_external<CRStride, CCStride, PRStride, PCStride>(
                ics: &nalgebra::VectorView<T, nalgebra::Const<3>, CRStride, CCStride>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Vector3<T>>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                let quaternion = cs_state.q.clone();

                let ecs_norot = $fn_ecs::<T, { $params.len() }, PRStride, PCStride>(
                    (ics[0].clone(), ics[1].clone(), ics[2].clone()),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                );

                Some(
                    quaternion.transform_vector(&ecs_norot)
                        + nalgebra::Vector3::new(cs_state.x.clone(), T::zero(), cs_state.z.clone()),
                )
            }

            fn transform_external_to_internal<CRStride, CCStride, PRStride, PCStride>(
                ecs: &nalgebra::VectorView<T, nalgebra::Const<3>, CRStride, CCStride>,
                params: &nalgebra::VectorView<
                    T,
                    nalgebra::Const<{ $params.len() }>,
                    PRStride,
                    PCStride,
                >,
                cs_state: &Self::CSST,
            ) -> Option<nalgebra::Vector3<T>>
            where
                CRStride: nalgebra::Dim,
                CCStride: nalgebra::Dim,
                PRStride: nalgebra::Dim,
                PCStride: nalgebra::Dim,
            {
                let quaternion = cs_state.q.clone();

                let ecs_norot = quaternion.conjugate().transform_vector(
                    &(ecs
                        - nalgebra::Vector3::new(
                            cs_state.x.clone(),
                            T::zero(),
                            cs_state.z.clone(),
                        )),
                );

                Some($fn_ics::<T, { $params.len() }, PRStride, PCStride>(
                    (
                        ecs_norot[0].clone(),
                        ecs_norot[1].clone(),
                        ecs_norot[2].clone(),
                    ),
                    &Self::PARAM_NAMES,
                    params,
                    cs_state,
                ))
            }
        }
    };
}

pub(crate) use impl_xcgm_geometry;
