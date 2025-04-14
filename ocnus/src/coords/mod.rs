//! Coordinate systems, geometries and state types.
//!
//! The [`OcnusCoords`] trait describes the coordinate systems, or *geometries*, that are used
//! within the **ocnus** framework to conveniently describe the physical systems of the employed
//! models. An implementation of the trait guarantuees the existance of bi-directional coordinate
//! transformation functions, and methods that compute the covariant and contravariant basis
//! vectors.
//!
//! For consistency, the coordinate systems are all not orthonormal. Therefore, one must properly
//! account for using co- and contravariant basis vectors. Simple geometries may nonetheless have
//! orthogonal basis vectors.
//!
//! Implemented coordinate systems / geometries:
//! - [`CCGeometry`] A circular-cylindrical geometry with internal coordinates (r, ϕ, z) for flux
//!   rope models.
//! - [`ECGeometry`] An elliptic-cylindrical geometry with internal coordinates (μ, ν, z) for flux
//!   rope models.
//! - [`TTGeometry`] A tapered-toroidal geometry with an elliptical cross-section and internal
//!   coordinates (μ, ν, s) for flux rope models.
//! - [`SPHGeometry`] A spherical geometry with internal coordiantes (r, ϕ, θ) for solar wind or
//!   spheromak models.
//!
//! Each geometry is associated with a coordinate system state type that allows for the defintion
//! of time-varying coordatinate systems. The coordinate system state types must be initialized
//! from the coordinate parameters using an implementation of [`OcnusCoords::initialize_cst`].
//!
//! Implemented coordinate system state types:
//! - [`XCState`] A generic coordinate system state type for cylindrical geometries with arbitrary
//!   cross-sections.
//! - [`TTState`] A coordinate system state type for [`TTGeometry`].
//! - [`SPHState`] A coordinate system state type for [`SPHGeometry`].

mod sphgm;
mod ttgm;
mod xcgm;

pub use sphgm::{SPHGeometry, SPHState};
pub use ttgm::{TTGeometry, TTState};
pub use xcgm::{CCGeometry, ECGeometry, XCState};

use crate::fXX;
use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use thiserror::Error;

/// Errors associated with the [`coords`](crate::coords) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum CoordsError {
    #[error("failed conversion from internal into external coordinates")]
    ExternalCoordsNotFound,
    #[error("failed conversion from external into internal coordinates")]
    InternalCoordsNotFound,
}

/// A trait that must be implemented for any type that represents a 3D curvilinear coordinate
/// system with `P` model parameters and a coordinate system state type `CSST`.
pub trait OcnusCoords<T, const P: usize, CSST>
where
    T: fXX,
{
    /// Static coordinate parameter names.
    const COORD_PARAMS: [&'static str; P];

    /// Computes the local contravariant basis vectors.
    fn contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CSST,
    ) -> Result<[Vector3<T>; 3], CoordsError>;

    /// Computes the local covariant basis vectors.
    fn covariant_basis<CStride: Dim>(
        _ics: &VectorView3<T>,
        _params: &VectorView<T, Const<P>, U1, CStride>,
        _state: &CSST,
    ) -> Result<[Vector3<T>; 3], CoordsError> {
        unimplemented!("covariant basis vectors are currently not implemented")
    }

    /// Create a vector from contravariant components.
    fn contravariant_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        components: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CSST,
    ) -> Result<Vector3<T>, CoordsError> {
        let basis = Self::contravariant_basis(ics, params, state)?;

        Ok(basis
            .iter()
            .zip(components.iter())
            .map(|(b, c)| b.clone() * *c)
            .sum())
    }

    /// Transform external coordinates `ecs` into the internal coordinates `ics`.
    fn transform_ics_to_ecs<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CSST,
    ) -> Result<Vector3<T>, CoordsError>;

    /// Transform internal coordinates `ics` into cartesian coordinates `ecs`.
    fn transform_ecs_to_ics<CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CSST,
    ) -> Result<Vector3<T>, CoordsError>;

    /// Initialize the coordinate state.
    fn initialize_cst<CStride: Dim>(
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &mut CSST,
    );

    /// Retrieve a model parameter index by name.
    fn param_index(name: &str) -> Option<usize> {
        Self::COORD_PARAMS
            .into_iter()
            .position(|param| param == name)
    }

    /// Retrieve a model parameter value by name.
    fn param_value<CStride: Dim>(
        name: &str,
        params: &VectorView<T, Const<P>, U1, CStride>,
    ) -> Option<T>
    where
        T: Clone,
    {
        if let Some(index) = Self::param_index(name) {
            // Explicit clone as T is a primitive.
            Some(params[index].clone())
        } else {
            None
        }
    }

    /// Test the implementation of the contravariant basis vectors.
    #[cfg(test)]
    fn test_contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,

        delta_h: T,
    ) where
        CSST: Default,
    {
        use crate::math::T;

        let mut state = CSST::default();

        Self::initialize_cst(&params.fixed_rows::<P>(0), &mut state);

        let ics_1p = ics + Vector3::<T>::x_axis().into_inner() * delta_h / T!(2.0);
        let ics_1m = ics - Vector3::<T>::x_axis().into_inner() * delta_h / T!(2.0);

        let ics_2p = ics + Vector3::<T>::y_axis().into_inner() * delta_h / T!(2.0);
        let ics_2m = ics - Vector3::<T>::y_axis().into_inner() * delta_h / T!(2.0);

        let ics_3p = ics + Vector3::<T>::z_axis().into_inner() * delta_h / T!(2.0);
        let ics_3m = ics - Vector3::<T>::z_axis().into_inner() * delta_h / T!(2.0);

        let basis =
            Self::contravariant_basis(&ics.as_view(), &params.fixed_rows::<P>(0), &state).unwrap();

        let ecs_1p =
            Self::transform_ics_to_ecs(&ics_1p.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        let ecs_1m =
            Self::transform_ics_to_ecs(&ics_1m.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        let ecs_2p =
            Self::transform_ics_to_ecs(&ics_2p.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        let ecs_2m =
            Self::transform_ics_to_ecs(&ics_2m.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        let ecs_3p =
            Self::transform_ics_to_ecs(&ics_3p.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        let ecs_3m =
            Self::transform_ics_to_ecs(&ics_3m.as_view(), &params.fixed_rows::<P>(0), &state)
                .unwrap();

        // dbg!(&(basis[0], (ecs_1p - ecs_1m) / delta_h));
        // dbg!(&(basis[1], (ecs_2p - ecs_2m) / delta_h));
        // dbg!(&(basis[2], (ecs_3p - ecs_3m) / delta_h));

        assert!((basis[0] - (ecs_1p - ecs_1m) / delta_h).norm() < T!(10.0) * delta_h);
        assert!((basis[1] - (ecs_2p - ecs_2m) / delta_h).norm() < T!(10.0) * delta_h);
        assert!((basis[2] - (ecs_3p - ecs_3m) / delta_h).norm() < T!(10.0) * delta_h);
    }
}
