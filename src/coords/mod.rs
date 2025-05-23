//! # Curvilinear coordinate systems and model geometries.
//!
//! This module introduces the [`OcnusCoords`] trait, which is the trait that is shared by all curvilinear coordinate systems that define a model geometry.
//! The trait provides bi-directional coordinate transformation functions, and methods that compute the covariant and contravariant basis vectors.
//! The basis vectors of the coordinate systems are not necessarily orthonormal. Therefore, one must properly account for using co- and contravariant basis vectors. Simple geometries may nonetheless have orthogonal basis vectors.

//! Currently implemented coordinate systems & model geometries:
//! - [`CCGeometry`] A circular-cylindrical geometry with internal coordinates (r, ϕ, z) for flux
//!   rope models.
//! - [`ECGeometry`] An elliptic-cylindrical geometry with internal coordinates (μ, ν, z) for flux
//!   rope models.
//! - [`TTGeometry`] A tapered-toroidal geometry with an elliptical cross-section and internal
//!   coordinates (μ, ν, ) for flux rope models.
//! - [`SPHGeometry`] A spherical geometry with internal coordiantes (r, ϕ, θ) for spheromak models.
//! - [`SPHUGeometry`] A unit sphere with internal coordiantes (r, ϕ, θ) for global heliospheric models.
//!
//! Each geometry is associated with a fixed coordinate system state type, which enables the description of time-varying coordatinate systems.
//! The coordinate system state types must be initialized from the coordinate system parameters using an implementation of [`OcnusCoords::initialize_cs`].

mod sphgm;
mod ttgm;
mod xcgm;

use std::iter::Sum;

pub use sphgm::{SPHGeometry, SPHState, SPHUGeometry};
pub use ttgm::{TTGeometry, TTState};
pub use xcgm::{CCGeometry, ECGeometry, XCState};

use nalgebra::{
    Const, Dim, RealField, SVector, SVectorView, Unit, UnitQuaternion, Vector3, VectorView,
    VectorView3,
};

/// A trait that is shared by all coordinate systems describing a model geometry.
///
/// Each coordinate system is associated with a single coordinate system state type `CSST`.
pub trait OcnusCoords<T, const D: usize, CSST>
where
    T: Copy + RealField,
    Self: Send + Sync,
{
    /// Coordinate system parameter names.
    const PARAMS: SVector<&'static str, D>;

    /// Coordinate system parameter name count.
    const PARAMS_COUNT: usize = D;

    /// Computes the local contravariant basis vectors.
    fn contravariant_basis<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<[Vector3<T>; 3]>;

    /// Computes the local contravariant basis vectors and returns the normalized vectors.
    fn contravariant_basis_normalized<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<[Vector3<T>; 3]> {
        let [dmu, dnu, ds] = Self::contravariant_basis(ics, params, cs_state)?;

        Some([dmu / dmu.norm(), dnu / dnu.norm(), ds / ds.norm()])
    }

    /// Computes the local covariant basis vectors.
    fn covariant_basis(
        _ics: &VectorView3<T>,
        _params: &SVectorView<T, D>,
        _state: &CSST,
    ) -> Option<[Vector3<T>; 3]> {
        unimplemented!("computation of the covariant basis vectors is currently not implemented")
    }

    /// Create a vector from contravariant components.
    fn contravariant_vector<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        components: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<Vector3<T>>
    where
        SVector<T, 3>: Sum,
    {
        let basis = Self::contravariant_basis(ics, params, cs_state)?;

        Some(
            basis
                .iter()
                .zip(components.iter())
                .map(|(basis, component)| basis * *component)
                .sum(),
        )
    }

    /// Create a vector from contravariant components, using the normalized basis vectors.
    fn contravariant_vector_normalized<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        components: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<Vector3<T>>
    where
        SVector<T, 3>: Sum,
    {
        let basis = Self::contravariant_basis_normalized(ics, params, cs_state)?;

        Some(
            basis
                .iter()
                .zip(components.iter())
                .map(|(basis, component)| basis * *component)
                .sum(),
        )
    }

    /// Compute the determinant of the metric tensor.
    fn detg<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<T>;

    /// Initialize the coordinate system state.
    ///
    /// This function may panic for invalid parameter inputs.
    fn initialize_cs<RStride: Dim, CStride: Dim>(
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &mut CSST,
    );

    /// Transform external coordinates `ecs` into the internal coordinates `ics`.
    fn transform_ics_to_ecs<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<Vector3<T>>;

    /// Transform internal coordinates `ics` into cartesian coordinates `ecs`.
    fn transform_ecs_to_ics<RStride: Dim, CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        cs_state: &CSST,
    ) -> Option<Vector3<T>>;

    /// Test the implemented trait functions.
    #[cfg(test)]
    fn test_implementation<RStride: Dim, CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<D>, RStride, CStride>,
        delta_h: T,
    ) where
        CSST: Default,
    {
        let mut cs_state = CSST::default();

        Self::initialize_cs(&params.as_view(), &mut cs_state);

        let ics_1p =
            ics + Vector3::<T>::x_axis().into_inner() * delta_h / T::from_usize(2).unwrap();
        let ics_1m =
            ics - Vector3::<T>::x_axis().into_inner() * delta_h / T::from_usize(2).unwrap();

        let ics_2p =
            ics + Vector3::<T>::y_axis().into_inner() * delta_h / T::from_usize(2).unwrap();
        let ics_2m =
            ics - Vector3::<T>::y_axis().into_inner() * delta_h / T::from_usize(2).unwrap();

        let ics_3p =
            ics + Vector3::<T>::z_axis().into_inner() * delta_h / T::from_usize(2).unwrap();
        let ics_3m =
            ics - Vector3::<T>::z_axis().into_inner() * delta_h / T::from_usize(2).unwrap();

        let basis = Self::contravariant_basis(
            &ics.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_1p = Self::transform_ics_to_ecs(
            &ics_1p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_1m = Self::transform_ics_to_ecs(
            &ics_1m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_2p = Self::transform_ics_to_ecs(
            &ics_2p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_2m = Self::transform_ics_to_ecs(
            &ics_2m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_3p = Self::transform_ics_to_ecs(
            &ics_3p.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        let ecs_3m = Self::transform_ics_to_ecs(
            &ics_3m.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        assert!(
            (basis[0] - (ecs_1p - ecs_1m) / delta_h).norm() < T::from_usize(10).unwrap() * delta_h
        );
        assert!(
            (basis[1] - (ecs_2p - ecs_2m) / delta_h).norm() < T::from_usize(10).unwrap() * delta_h
        );
        assert!(
            (basis[2] - (ecs_3p - ecs_3m) / delta_h).norm() < T::from_usize(10).unwrap() * delta_h
        );

        let detg_basis = (basis[0].cross(&basis[1]).dot(&basis[2])).abs();
        let detg_analy = Self::detg(
            &ics.as_view(),
            &params.rows_generic(0, params.shape_generic().0),
            &cs_state,
        )
        .unwrap();

        assert!(approx::ulps_eq!(detg_basis, detg_analy));
    }
}

/// Retrieve a model parameter index by name.
pub fn param_index<const D: usize>(name: &str, names: &SVector<&'static str, D>) -> Option<usize> {
    names.iter().position(|param| *param == name)
}

/// Retrieve a model parameter value by name.
pub fn param_value<T, const D: usize, RStride: Dim, CStride: Dim>(
    name: &str,
    names: &SVector<&'static str, D>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
) -> Option<T>
where
    T: Copy,
{
    param_index(name, names).map(|index| params[index])
}

/// Generate unit quaternion from three successive rotations around the z, y and x-axis.
pub fn quaternion_rot<T>(z_angle: T, y_angle: T, x_angle: T) -> UnitQuaternion<T>
where
    T: Copy + RealField,
{
    let uz = Vector3::<T>::z_axis();
    let uy = Vector3::<T>::y_axis();
    let ux = Vector3::<T>::x_axis();

    let rot_z = UnitQuaternion::from_axis_angle(&uz, z_angle);

    let rot_y = UnitQuaternion::from_axis_angle(
        &Unit::new_unchecked(rot_z.transform_vector(&uy)),
        -y_angle,
    );

    let rot_x = UnitQuaternion::from_axis_angle(
        &Unit::new_unchecked((rot_y * rot_z).transform_vector(&ux)),
        x_angle,
    );

    rot_x * (rot_y * rot_z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternions() {
        assert!(quaternion_rot(0.0, 0.0, 0.0) == UnitQuaternion::identity());

        let rad90 = 90.0_f64.to_radians();

        assert!(
            quaternion_rot(rad90, 0.0, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, 0.0, 1.0) * rad90)
        );
        assert!(
            quaternion_rot(0.0, rad90, 0.0)
                == UnitQuaternion::new(Vector3::new(0.0, -1.0, 0.0) * rad90)
        );
        assert!(
            quaternion_rot(0.0, 0.0, rad90)
                == UnitQuaternion::new(Vector3::new(1.0, 0.0, 0.0) * rad90)
        );
    }
}
