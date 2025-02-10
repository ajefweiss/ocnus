use nalgebra::SVectorView;
use serde::Serialize;
use thiserror::Error;

use crate::{fevms::FEVMError, stats::OcnusStatsError};

/// Generic error type returned by types that implement [`OcnusModel`].
#[derive(Debug, Error)]
pub enum OcnusError {
    #[error(transparent)]
    FEVM(FEVMError),
    #[error(transparent)]
    Stats(#[from] OcnusStatsError),
}

/// A trait that must be implemented for any type that acts as a generic model.
///
/// This trait, by itself, does not provide much useful functionality and only provides access
/// to the model parameters and the valid parameter range.
pub trait OcnusModel<const P: usize> {
    /// Const array of parameter names.
    const PARAMS: [&'static str; P];

    /// Const array of parameter ranges.
    const PARAM_RANGES: [(f32, f32); P];

    /// Get a model parameter index by name.
    fn get_param_index(name: &str) -> Option<usize> {
        Self::PARAMS.into_iter().position(|param| param == name)
    }

    /// Get a model parameter value by name.
    fn get_param_value(name: &str, params: &SVectorView<f32, P>) -> Option<f32> {
        Some(params[Self::get_param_index(name)?])
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

// Implement the [`OcnusObser`] traits for Fp.
impl OcnusObser for f32 {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + Serialize + Sync {}

// Implement the [`OcnusState`] traits for Fp.
impl OcnusState for f32 {}
