//! Coordinate systems.
//!
//! The [`OcnusCoords`] trait describes the coordinate systems, or *geometries*, that are used
//! within the **ocnus** framework to conveniently describe the physical systems of the employed
//! models. An implementation of the trait guarantuees the existance of bi-directional coordinate
//! transformation functions, and methods that compute the covariant and contravariant basis
//! vectors.
//!
//! Implemented coordinate systems / geometries:
//! - [`CCGeometry`] A circular-cylindrical geometry with internal coordinates (r, ϕ, z) for flux
//!   rope models.
//! - [`ECGeometry`] An elliptic-cylindrical geometry with internal coordinates (r, ϕ, z) for flux
//!   rope models.
//! - [`SPHGeometry`] A spherical geometry with internal coordiantes (r, ϕ, θ) for solar wind or
//!   spheromak models.
//!
//! #### States
//!
//! The geometry can either depend on model parameters or on a state variable `state` that is
//! introduced to allow for time-dependence. Each geometry must make use of a state, although
//! multiple geometries can share the same type.
//!
//! State types:
//! - [`XCState`] A generic state for cylindrical geometries with arbitrary cross-sections.
//! - [`SPHState`] A state for spherical geometries.

mod sphgm;
mod ttgm;
mod xcgm;

// pub use sphgm::{SPHGeometry, SPHState};
pub use ttgm::{TTGeometry, TTState};
pub use xcgm::{CCGeometry, ECGeometry, XCState};

use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use thiserror::Error;

use crate::fXX;

/// Errors associated with the [`coords`](crate::coords) module.
#[allow(missing_docs)]
#[derive(Debug, Error)]
pub enum CoordsError {
    #[error("failed conversion from internal into external coordinates")]
    ExternalCoordsNotFound,
    #[error("failed conversion from external into internal coordinates")]
    InternalCoordsNotFound,
}

/// A trait that must be implemented for any type that represents a 3D curvilinear coordinate system.
pub trait OcnusCoords<T, const P: usize, CST>
where
    T: fXX,
{
    /// Static coordinate parameter names.
    const COORD_PARAMS: [&'static str; P];

    /// Computes the local contravariant basis vectors.
    fn contravariant_basis<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CST,
    ) -> Result<[Vector3<T>; 3], CoordsError>;

    /// Computes the local covariant basis vectors.
    fn covariant_basis<CStride: Dim>(
        _ics: &VectorView3<T>,
        _params: &VectorView<T, Const<P>, U1, CStride>,
        _state: &CST,
    ) -> Result<[Vector3<T>; 3], CoordsError> {
        unimplemented!("covariant basis vectors are currently not implemented")
    }

    /// Create a vector from contravariant components.
    fn contravariant_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        components: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CST,
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
        state: &CST,
    ) -> Result<Vector3<T>, CoordsError>;

    /// Transform internal coordinates `ics` into cartesian coordinates `ecs`.
    fn transform_ecs_to_ics<CStride: Dim>(
        ecs: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        state: &CST,
    ) -> Result<Vector3<T>, CoordsError>;

    /// Initialize the coordinate state.
    fn initialize_cst<CStride: Dim>(params: &VectorView<T, Const<P>, U1, CStride>, state: &mut CST);

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
}
