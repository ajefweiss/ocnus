// use crate::{
//     coords::OcnusCoords,
//     fXX,
//     math::{acos, atan2, cos, sin},
// };
// use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
// use serde::{Deserialize, Serialize};
// use std::{fmt::Debug, marker::PhantomData};

// /// Model state for a spherical geometry
// #[derive(Clone, Debug, Default, Deserialize, Serialize)]
// pub struct SPHState<T>
// where
//     T: fXX,
// {
//     /// Spherical center.
//     pub center: Vector3<T>,

//     /// Radial scale factor
//     pub radius: T,
// }

// /// Spherical geometry with arbitrary center position and radius.
// pub struct SPHGeometry<T>(PhantomData<T>)
// where
//     T: fXX;

// impl<T> Default for SPHGeometry<T>
// where
//     T: fXX,
// {
//     fn default() -> Self {
//         Self(PhantomData::<T>)
//     }
// }

// impl<T> OcnusCoords<T, 4, SPHState<T>> for SPHGeometry<T>
// where
//     T: fXX,
// {
//     const COORD_PARAMS: [&'static str; 4] = ["center_x0", "center_y0", "center_z0", "radius"];

//     fn contravariant_basis<CStride: Dim>(
//         ics: &VectorView3<T>,
//         _params: &VectorView<T, Const<4>, U1, CStride>,
//         _state: &SPHState<T>,
//     ) -> [Vector3<T>; 3] {
//         let phi = ics[1];
//         let theta = ics[2];

//         [
//             Vector3::new(
//                 sin!(theta) * cos!(phi),
//                 sin!(theta) * sin!(phi),
//                 cos!(theta),
//             ),
//             Vector3::new(-sin!(phi), cos!(phi), T::zero()),
//             Vector3::new(
//                 cos!(theta) * cos!(phi),
//                 cos!(theta) * sin!(phi),
//                 cos!(theta),
//             ),
//         ]
//     }

//     fn transform_ics_to_ecs<CStride: Dim>(
//         ics: &VectorView3<T>,
//         _params: &VectorView<T, Const<4>, U1, CStride>,
//         state: &SPHState<T>,
//     ) -> Vector3<T> {
//         let center = state.center;
//         let radius = state.radius;

//         let r = ics[0];
//         let phi = ics[1];
//         let theta = ics[2];

//         Vector3::new(
//             radius * r * cos!(phi) * sin!(theta),
//             radius * r * sin!(phi) * sin!(theta),
//             radius * r * cos!(theta),
//         ) + center
//     }

//     fn transform_ecs_to_ics<CStride: Dim>(
//         ecs: &VectorView3<T>,
//         _params: &VectorView<T, Const<4>, U1, CStride>,
//         state: &SPHState<T>,
//     ) -> Vector3<T> {
//         let center = state.center;
//         let radius = state.radius;

//         let v = ecs - center;
//         let vn = v.norm();

//         Vector3::new(vn / radius, acos!(v[2] / vn), atan2!(v[1], v[0]))
//     }

//     fn initialize_cst<CStride: Dim>(
//         params: &VectorView<T, Const<4>, U1, CStride>,
//         state: &mut SPHState<T>,
//     ) {
//         let x0 = Self::param_value("center_x0", params).unwrap();
//         let y0 = Self::param_value("center_y0", params).unwrap();
//         let z0 = Self::param_value("center_z0", params).unwrap();
//         let radius = Self::param_value("radius", params).unwrap();

//         state.center = Vector3::new(x0, y0, z0);
//         state.radius = radius;
//     }
// }
