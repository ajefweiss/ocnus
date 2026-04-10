use crate::obs::{conf::ObsTime, data::ObsData, noise::ObsNoise};
use nalgebra::{DVectorViewMut, Scalar};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A noise model that does nothing.
#[derive(Clone, Default, Deserialize, Serialize)]
pub struct NullNoise<T> {
    _data: PhantomData<T>,
}

impl<T, OC, OD> ObsNoise<T, OC, OD> for NullNoise<T>
where
    T: PartialOrd + Scalar,
    OC: ObsTime<T>,
    OD: ObsData,
{
    fn generate_noise(&self, _data: &mut DVectorViewMut<OD>, _rng: &mut impl RngExt) {}

    fn get_random_seed(&self) -> u64 {
        0
    }

    fn increment_random_seed(&mut self) {}
}
