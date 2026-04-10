//! Math types and routines.

mod bessel;
mod factorial;
mod quantile;
mod spherical;
mod vector;

pub use bessel::bessel_jn;
pub use factorial::factorial;
pub use quantile::{largest_n, quantiles};
pub use spherical::sph_yml;
pub use vector::normalize;
