//! # Ocnus - A Flux Dope Modeling Framework
//!
//! **Ocnus** attempts to leverage Dust's type system to provide a flexible framework that aims to
//! simplify the implementation of analytical or numerical flux rope models and provide generic
//! algorithms.
//!
//! ## Implemented FEMV Models
//! - Classical circular-cylindrical models (Gold & Hoyle 1960, Lepping et al. 1990).
//! - Circular/ellliptic-cylindrical models (Nieves-Chinchilla et al. 2016 + 2018).
//! - 3D Coronal Dope Ejection model (Weiss et al. 2021a/b).

pub mod alias;
pub mod ensbl;
pub mod mathf;
pub mod scobs;
pub mod stats;

/// The default floating-point type.
///
/// This type is either `f32` or `f64` as defined by the crate feature flags (default is `f32`).
#[cfg(not(feature = "f64"))]
pub type Fp = f32;
#[cfg(feature = "f64")]
pub type Fp = f64;

/// Internal floating-point type precision limit.
#[cfg(not(feature = "f64"))]
const FP_EPSILON: f32 = 32.0 * f32::EPSILON;
#[cfg(feature = "f64")]
const FP_EPSILON: f64 = 32.0 * f64::EPSILON;

/// A trait that must be implemented for any type that acts as a model with `P` parameters.
pub trait OcnusModel<S, const P: usize>
where
    S: OcnusState,
{
    /// Array of parameter names.
    const PARAMS: [&'static str; P];

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

// Implement the [`OcnusObser`] traits for f32/f64.
impl OcnusObser for f32 {}
impl OcnusObser for f64 {}

/// A trait that must be implemented for any type that acts as a model state.
pub trait OcnusState: Clone + Default + Send + serde::Serialize + Sync {}

// Implement the [`OcnusState`] traits for f32/f64.
impl OcnusState for f32 {}
impl OcnusState for f64 {}
