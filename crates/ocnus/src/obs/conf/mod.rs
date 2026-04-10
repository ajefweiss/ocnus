//! Observation configuration types.

mod cam;
mod pos;

use nalgebra::{SVector, Scalar};
use wcs::WCSParams;

pub use cam::*;
pub use pos::*;

/// A trait that must be implemented by all observation configuration types.
pub trait ObsTime<T>: Scalar
where
    T: Scalar,
{
    /// Returns the timestamp of the observation.
    fn timestamp(&self) -> T;
}

/// A trait for observation configurations providing position information.
pub trait ObsPosition<T, const D: usize>: ObsTime<T>
where
    T: Scalar,
{
    /// Returns the position of the observation.
    fn position(&self) -> SVector<T, D>;
}

/// A trait for observation configurations providing remote observations in terms of WCS parameters.
pub trait ObsCam<T>: ObsPosition<T, 3>
where
    T: Scalar,
{
    /// Returns the polarization angle w.r.t. the x-axis of the image plane, if available.
    fn polarization(&self) -> Option<T>;

    /// Returns the WCS parameters of the remote observation.
    fn wcs(&self) -> &WCSParams;
}
