use crate::Fp;

/// A trait that must be implemented for any type that acts as a model within the framework.
pub trait OcnusModel<const P: usize> {
    /// Array of parameter names.
    const PARAMS: [&'static str; P];

    /// Array of parameter bounds.
    const PARAM_BOUNDS: [(f32, f32); P];

    /// Compute adaptive step sizes for finite differences based on the valid range of the model prior.
    ///
    /// The step size for constant parameters is zero.
    fn adaptive_step_sizes(&self) -> [f32; P] {
        self.valid_range()
            .iter()
            .map(|(min, max)| 1024.0 * (max - min) * f32::EPSILON)
            .collect::<Vec<f32>>()
            .try_into()
            .unwrap()
    }

    /// Get parameter index by name.
    fn get_param_index(name: &str) -> Option<usize> {
        Self::PARAMS.into_iter().position(|param| param == name)
    }

    /// Returns the valid range for parameter vector samples.
    fn valid_range(&self) -> [(Fp, Fp); P];
}

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + serde::Serialize + Sync {}

// Implement the [`OcnusObser`] traits for Fp.
impl OcnusObser for Fp {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + serde::Serialize + Sync {}

// Implement the [`OcnusState`] traits for Fp.
impl OcnusState for Fp {}
