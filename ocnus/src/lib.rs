#![doc = include_str!("../../README.md")]
#![deny(missing_docs)]

pub mod fevm;
pub mod geometry;
pub mod math;
pub mod obser;
mod scobs;
pub mod stats;

pub use scobs::*;

use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};
use serde::Serialize;

/// A trait that must be implemented for any type that acts as a generic model.
///
/// This trait, by itself, does not provide much useful functionality and only provides access
/// to the model parameters and the valid parameter range.
pub trait OcnusModel<const P: usize, S>
where
    S: OcnusState,
{
    /// Const array of parameter nwames.
    const PARAMS: [&'static str; P];

    /// Const array of parameter ranges.
    const PARAM_RANGES: [(f32, f32); P];

    /// Returns the intrinsic basis vectors as a function of intrinsic coordinates.
    fn basis_from_ics<CStride: Dim>(
        ics: &VectorView3<f32>,
        vec: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &S,
    ) -> Vector3<f32>;

    /// Compute the basis vectors in the intrinsic coordinate system.
    fn coords_basis_ics<CStride: Dim>(
        ics: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &S,
    ) -> [Vector3<f32>; 3];

    /// Transform intrinsic coordinates `ics` into external cartesian coordinates.
    fn coords_from_ics<CStride: Dim>(
        ics: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &S,
    ) -> Vector3<f32>;

    /// Transform cartensian coordinates `xyz` into the intrinsic coordinates.
    fn coords_into_ics<CStride: Dim>(
        xyz: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &S,
    ) -> Vector3<f32>;

    /// Get a model parameter index by name.
    fn get_param_index(name: &str) -> usize {
        Self::PARAMS
            .into_iter()
            .position(|param| param == name)
            .unwrap_or_else(|| panic!("no \"{}\" parameter", name))
    }

    /// Get a model parameter value by name.
    fn get_param_value<CStride: Dim>(
        name: &str,
        params: &VectorView<f32, Const<P>, U1, CStride>,
    ) -> f32 {
        params[Self::get_param_index(name)]
    }

    /// Compute adaptive step sizes for finite difference calculations for each model parameter
    /// based on the valid range.
    ///
    /// Note that the step size for constant parameters is zero.
    fn param_step_sizes(&self) -> [f32; P] {
        Self::PARAM_RANGES
            .iter()
            .map(|(min, max)| 16.0 * (max - min) * f32::EPSILON)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap()
    }
}

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + Serialize + Sync {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + Serialize + Sync {}
