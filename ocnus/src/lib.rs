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
pub const FP_EPSILON: f32 = 32.0 * f32::EPSILON;
#[cfg(feature = "f64")]
pub const FP_EPSILON: f64 = 32.0 * f64::EPSILON;
