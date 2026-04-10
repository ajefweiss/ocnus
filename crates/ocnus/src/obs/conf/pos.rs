use crate::obs::conf::{ObsPosition, ObsTime};
use nalgebra::{SVector, Scalar};
use serde::{Deserialize, Serialize};

/// A struct for storing a simple spacecraft observation configuration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct VecConf<T, const D: usize>
where
    T: Scalar,
{
    timestamp: T,
    position: SVector<T, D>,
}

impl<T, const D: usize> VecConf<T, D>
where
    T: Scalar,
{
    /// Creates a new [`VecConf`] object.
    pub fn new(timestamp: T, position: SVector<T, D>) -> Self {
        Self {
            timestamp,
            position,
        }
    }
}

impl<T, const D: usize> From<(T, SVector<T, D>)> for VecConf<T, D>
where
    T: Scalar,
{
    fn from(value: (T, SVector<T, D>)) -> Self {
        Self {
            timestamp: value.0,
            position: value.1,
        }
    }
}

impl<T, const D: usize> ObsTime<T> for VecConf<T, D>
where
    T: PartialEq + Scalar,
{
    fn timestamp(&self) -> T {
        self.timestamp.clone()
    }
}

impl<T, const D: usize> ObsPosition<T, D> for VecConf<T, D>
where
    T: Scalar,
{
    fn position(&self) -> SVector<T, D> {
        self.position.clone()
    }
}
