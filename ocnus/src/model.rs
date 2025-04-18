use crate::Fp;
use nalgebra::SVectorView;
use serde::Serialize;

/// A trait that must be implemented for any type that acts as a model within the framework.
pub trait OcnusModel<const P: usize> {
    /// Array of parameter names.
    const PARAMS: [&'static str; P];

    /// Compute adaptive step sizes for finite differences based on the valid range of the model prior.
    ///
    /// The step size for constant parameters is zero.
    fn adaptive_step_sizes(&self) -> [Fp; P] {
        self.valid_range()
            .iter()
            .map(|(min, max)| 256.0 * (max - min) * Fp::EPSILON)
            .collect::<Vec<Fp>>()
            .try_into()
            .unwrap()
    }

    /// Get parameter index by name.
    fn get_param_index(name: &str) -> Option<usize> {
        Self::PARAMS.into_iter().position(|param| param == name)
    }

    /// Get parameter value by name.
    fn get_param_value(name: &str, params: &SVectorView<Fp, P>) -> Option<Fp> {
        Some(params[Self::get_param_index(name)?])
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); P];
}

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + Serialize + Sync {}

// Implement the [`OcnusObser`] traits for Fp.
impl OcnusObser for Fp {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + Serialize + Sync {}

// Implement the [`OcnusState`] traits for Fp.
impl OcnusState for Fp {}
