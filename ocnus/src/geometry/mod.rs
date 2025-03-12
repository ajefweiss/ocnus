//! Geometry models, i.e. , i.e. implementations of [`OcnusGeometry`].

mod acylm;

pub use acylm::*;

use nalgebra::{Const, Dim, U1, Vector3, VectorView, VectorView3};

/// A trait that must be implemented for any type that acts as a model geometry.
///
/// This trait, by itself, does not provide much useful functionality and only provides access
/// to the model parameters, coordinate transformations and the respective valid parameter range.
pub trait OcnusGeometry<const P: usize, GS> {
    /// Const array of parameter nwames.
    const PARAMS: [&'static str; P];

    /// Const array of parameter ranges.
    const PARAM_RANGES: [(f32, f32); P];

    /// Compute the basis vectors in cartensian coordinates `xyz`
    /// as a function of intrinsic coordinates.
    ///
    /// These basis vectors are NOT necessarily normalized.
    fn coords_basis<CStride: Dim>(
        ics: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &GS,
    ) -> [Vector3<f32>; 3];

    /// Transform cartensian coordinates `xyz` into the intrinsic coordinates.
    fn coords_ics<CStride: Dim>(
        xyz: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &GS,
    ) -> Vector3<f32>;

    /// Transform intrinsic coordinates `ics` into external cartesian coordinates.
    fn coords_xyz<CStride: Dim>(
        ics: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &GS,
    ) -> Vector3<f32>;

    /// Transform a vector at/in intrinsic coordinates `ics` into external coordinates.
    fn coords_xyz_vector<CStride: Dim>(
        ics: &VectorView3<f32>,
        vec: &VectorView3<f32>,
        params: &VectorView<f32, Const<P>, U1, CStride>,
        state: &GS,
    ) -> Vector3<f32>;

    /// Retrieve a model parameter index by name.
    fn param_index(name: &str) -> usize {
        Self::PARAMS
            .into_iter()
            .position(|param| param == name)
            .unwrap_or_else(|| panic!("no \"{}\" parameter", name))
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

    /// Retrieve a model parameter value by name.
    fn param_value<CStride: Dim>(
        name: &str,
        params: &VectorView<f32, Const<P>, U1, CStride>,
    ) -> f32 {
        params[Self::param_index(name)]
    }
}
