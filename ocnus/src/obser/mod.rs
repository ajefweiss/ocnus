//! Observable types, i.e. implementations of [`OcnusObser`].

mod vector;

pub use vector::ObserVec;

use serde::Serialize;

/// A trait that must be implemented for any type that acts as a model observable.
pub trait OcnusObser: Clone + Default + Send + Serialize + Sync {}
