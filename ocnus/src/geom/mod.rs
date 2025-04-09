//! Geometry models and states.
//!
//! # Geometry
//!
//! A geometry model defines the basis of any more complex model within the **ocnus** framework
//! and defines the underlying internal coordinate system (ics) which is being used to
//! conveniently describe the physical system. The [`OcnusGeometry`] trait guarantuees the
//! existance of bi-directional coordinate transformations and also requires methods to
//! compute the basis vectors within the overarching Cartesian coordinate system.
//!
//! Default geometries:
//! - [`CCGeometry`] A circular-cylindrical geometry with internal coordinates (r, ϕ, z)
//!   for flux rope models.
//! - [`ECGeometry`] An elliptic-cylindrical geometry with internal coordinates (r, ϕ, z)
//!   for flux rope models.
//! - [`SPHGeometry`] A spherical geometry with internal coordiantes (r, ϕ, θ) for solar wind
//!   or spheromak models.
//!
//! #### States
//!
//! The geometry can either depend on model parameters or on a state variable `geom_state`
//! that is introduced to allow for time-dependence. Each geometry must make use of a state,
//! although multiple geometries can share the same type.
//!
//! Default states:
//! - [`XCState`] A generic state for cylindrical geometries with arbitrary cross-sections.
//! - [`SPHState`] A state for spherical geometries.

mod sphgm;
// mod ttgm;
mod xcgm;

pub use sphgm::{SPHGeometry, SPHState};
pub use xcgm::{CCGeometry, ECGeometry, XCState};

use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};

/// A trait that must be implemented for any type that acts as a model geometry.
///
/// This trait, by itself, is not intended to provide any useful functionality.
/// It only provides access to the model parameters, coordinate transformations
/// and the respective valid parameter range.
pub trait OcnusGeometry<T, const P: usize, GS> {
    /// Const array of parameter nwames.
    const PARAMS: [&'static str; P];

    /// Compute the basis vectors in cartesian coordinates
    /// as a function of intrinsic coordinates.
    ///
    /// Note: These basis vectors are not necessarily orthonormal.
    fn basis_vectors<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        geom_state: &GS,
    ) -> [Vector3<T>; 3];

    /// Transform cartesian coordinates `xyz` into the intrinsic coordinates `ics`.
    fn coords_ics_to_xyz<CStride: Dim>(
        ics: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        geom_state: &GS,
    ) -> Vector3<T>;

    /// Transform intrinsic coordinates `ics` into cartesian coordinates `xyz`.
    fn coords_xyz_to_ics<CStride: Dim>(
        xyz: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        geom_state: &GS,
    ) -> Vector3<T>;

    /// Create a new vector in cartesian coordinates given intrinsic vector components
    /// at a specific location given by intrinsic coordiantes `ics`.
    fn create_xyz_vector<CStride: Dim>(
        ics: &VectorView3<T>,
        ics_comp: &VectorView3<T>,
        params: &VectorView<T, Const<P>, U1, CStride>,
        geom_state: &GS,
    ) -> Vector3<T>;

    /// Initialize the geometry state for given model parameters.
    fn geom_state<CStride: Dim>(params: &VectorView<T, Const<P>, U1, CStride>, geom_state: &mut GS);

    /// Retrieve a model parameter index by name.
    fn param_index(name: &str) -> usize {
        Self::PARAMS
            .into_iter()
            .position(|param| param == name)
            .unwrap_or_else(|| panic!("no \"{}\" parameter", name))
    }

    /// Retrieve a model parameter value by name.
    fn param_value<CStride: Dim>(name: &str, params: &VectorView<T, Const<P>, U1, CStride>) -> T
    where
        T: Clone,
    {
        // Allow explicit clone as T is a primitive.
        params[Self::param_index(name)].clone()
    }
}
