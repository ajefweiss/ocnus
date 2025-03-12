// STATUS: Mature
// TODO: Add more ScObsConf enum variants for more complicated configurations (remote sensing etc).

use derive_more::IntoIterator;
use itertools::zip_eq;
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, max},
    ops::{Add, AddAssign},
};

/// The configuration of a single spacecraft observation, as used in [`ScObs`].
///
/// The following variants are currently implemented:
/// - [`ScObsConf::Distance`] : (x) - position of the spacecraft in a heliocentric coordinate
///   system.
/// - [`ScObsConf::Position`] : (x, y, z) - position of the spacecraft in a heliocentric coordinate
///   system.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ScObsConf {
    /// Distance from the Sun.
    Distance(f32),
    /// Position in space, in an arbitrary Solar centric coordiante system.
    Position([f32; 3]),
}

/// Represents a single spacecraft observation in time, with an optional observation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScObs<O> {
    configuration: ScObsConf,
    timestamp: f32,
    observation: Option<O>,
}

impl<O> ScObs<O> {
    /// Acces the configuration field.
    pub fn configuration(&self) -> &ScObsConf {
        &self.configuration
    }

    /// Create a new [`ScObs`].
    pub fn new(timestamp: f32, configuration: ScObsConf, opt_observation: Option<O>) -> Self {
        Self {
            configuration,
            timestamp,
            observation: opt_observation,
        }
    }

    /// Acces the timestamp field.
    pub fn timestamp(&self) -> f32 {
        self.timestamp
    }

    /// Acces the observation field.
    pub fn observation(&self) -> Option<&O> {
        self.observation.as_ref()
    }
}

/// Represents a time series of spacecraft observations, with optional observations.
///
/// [`ScObsSeries`] has, among others, three important implementations:
/// - [`ScObsSeries::add`] : Allows composition of two [`ScObsSeries`] objects.
/// - [`ScObsSeries::sort_by_timestamp`] : Sorts the underlying vector of [`ScObs`]
///   objets by their timestamps.
/// - [`ScObsSeries::split`] : The reciprocal of one or multiple [`ScObsSeries::add`] calls.
///   Calling this function consumes a composite [`ScObsSeries`] and returns the original
///   [`ScObsSeries`] objects in a vector.
#[derive(Clone, Debug, Deserialize, IntoIterator, Serialize)]
pub struct ScObsSeries<O> {
    /// Vector of spacecraft observations.
    #[into_iterator(ref)]
    scobs: Vec<ScObs<O>>,

    /// The sorting indices that are used to recover the original [`ScObsSeries`]
    /// objects from a composite.
    sorti: Vec<usize>,
}

impl<O> ScObsSeries<O> {
    /// Returns the number of individual [`ScObsSeries`] contained within.
    ///
    /// This value corresponds to the length of the vector returned by
    /// [`ScObsSeries::sort_by_timestamp`].
    pub fn count_series(&self) -> usize {
        self.sorti.iter().fold(0, |acc, next| max(acc, *next)) + 1
    }

    /// Create a [`ScObsSeries`] from an iterator over `ScObs`.
    pub fn from_iterator<I: IntoIterator<Item = ScObs<O>>>(iter: I) -> Self {
        let scobs = iter.into_iter().collect::<Vec<ScObs<O>>>();
        let length = scobs.len();

        Self {
            scobs,
            sorti: vec![0; length],
        }
    }

    /// Returns `true`` if the observation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.scobs.is_empty()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.scobs.len()
    }

    /// Sorts the underlying `Vec<ScObs>` object by the time stamp field.
    pub fn sort_by_timestamp(&mut self) {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort_closure = |scobs: &mut Vec<ScObs<O>>, sorti: &mut Vec<usize>| {
            let mut counter = 0;

            for idx in 0..(scobs.len() - 1) {
                if scobs[idx].timestamp > scobs[idx + 1].timestamp {
                    scobs.swap(idx, idx + 1);
                    sorti.swap(idx, idx + 1);

                    counter += 1
                }
            }

            counter
        };

        let mut counter = 1;

        while counter != 0 {
            counter = bubble_sort_closure(&mut self.scobs, &mut self.sorti);
        }
    }

    /// The reciprocal of [`ScObsSeries::sort_by_timestamp`].
    /// Calling this function consumes the [`ScObsSeries`] object and returns the original
    /// [`ScObsSeries`] objects in a vector.
    pub fn split(self) -> Vec<ScObsSeries<O>>
    where
        O: Clone,
    {
        (0..self.count_series())
            .map(|idx| {
                let copy_scobs = zip_eq(&self.scobs, &self.sorti)
                    .filter_map(|(obs, sdx)| match sdx.cmp(&idx) {
                        Ordering::Equal => Some(obs.clone()),
                        _ => None,
                    })
                    .collect::<Vec<ScObs<O>>>();

                let length = copy_scobs.len();

                Self {
                    scobs: copy_scobs,
                    sorti: vec![0; length],
                }
            })
            .collect()
    }
}

impl<O> Add for ScObsSeries<O> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut scobs = self.scobs;

        debug!(
            "merging two ScObs objects ({} + {})",
            scobs.len(),
            rhs.scobs.len()
        );

        scobs.extend(rhs.scobs);

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |acc: usize, &v| max(acc, v)) + 1;

        let mut sorti = self.sorti;

        // Add index_offset to all indices in rhs.
        sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        Self { scobs, sorti }
    }
}

impl<O> AddAssign for ScObsSeries<O> {
    fn add_assign(&mut self, rhs: Self) {
        debug!(
            "merging two ScObs objects ({} + {})",
            self.scobs.len(),
            rhs.scobs.len()
        );

        self.scobs.extend(rhs.scobs);

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |acc, &v| max(acc, v)) + 1;

        // Add index_offset to all indices in rhs.
        self.sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scobs_slice() {
        let sctc1 = ScObs {
            timestamp: 0.0,
            configuration: ScObsConf::Distance(1.0),
            observation: None::<f32>,
        };

        let sctc2 = ScObs {
            timestamp: 1.0,
            configuration: ScObsConf::Distance(1.0),
            observation: None::<f32>,
        };

        let sctc3 = ScObs {
            timestamp: 0.5,
            configuration: ScObsConf::Distance(1.0),
            observation: None::<f32>,
        };

        let mut ts3 = ScObsSeries::from_iterator([sctc1, sctc2]);
        let ts4 = ScObsSeries::from_iterator([sctc3]);

        assert!(ts3.count_series() == 1);
        assert!(ts4.count_series() == 1);
        assert!((&ts4).into_iter().last().unwrap().timestamp == 0.5);

        ts3 += ts4.clone();

        assert!(ts3.count_series() == 2);
        assert!(ts3.len() == 3);
        assert!(!ts3.is_empty());

        ts3.sort_by_timestamp();

        assert!(ts3.scobs.first().unwrap().timestamp == 0.0);
        assert!(ts3.scobs.last().unwrap().timestamp == 1.0);

        let unsort = ts3.clone().split();

        assert!(unsort.len() == 2);
        assert!(unsort[0].scobs.first().unwrap().timestamp == 0.0);
        assert!(unsort[0].scobs.last().unwrap().timestamp == 1.0);

        assert!(unsort[1].scobs.first().unwrap().timestamp == 0.5);

        let mut ts5 = ts3 + ts4;

        ts5.sort_by_timestamp();

        assert!(ts5.scobs.first().unwrap().timestamp == 0.0);
        assert!(ts5.scobs.last().unwrap().timestamp == 1.0);
    }
}
