use crate::{fXX, math::abs, obser::OcnusObser};
use derive_more::IntoIterator;
use itertools::zip_eq;
use log::debug;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, max},
    ops::{Add, AddAssign},
};

/// The configuration of a single spacecraft observation, as used in [`ScObs`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ScObsConf<T> {
    /// Distance from the Sun.
    Distance(T),
    /// Position in space, in an arbitrary Solar centric coordiante system.
    Position([T; 3]),
}

/// Represents a single spacecraft observation in time, with an optional observation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScObs<T, O> {
    configuration: ScObsConf<T>,
    observation: O,
    timestamp: T,
}

impl<T, O> ScObs<T, O> {
    /// Access the configuration field.
    pub fn configuration(&self) -> &ScObsConf<T> {
        &self.configuration
    }

    /// Compute distance betweeo two [`ScObs`]
    pub fn distance(&self, other: &Self) -> T
    where
        T: fXX,
    {
        let conf_0 = &self.configuration;
        let conf_1 = &other.configuration;

        match (conf_0, conf_1) {
            (ScObsConf::Distance(x_0), ScObsConf::Distance(x_1)) => abs!(*x_1 - *x_0),
            (ScObsConf::Position(r_0), ScObsConf::Position(r_1)) => {
                (Vector3::from(*r_1) - Vector3::from(*r_0)).norm()
            }
            _ => panic!("scobs configurations do not match"),
        }
    }

    /// Access the observation field.
    pub fn get_observation(&self) -> &O {
        &self.observation
    }

    /// Access the timestamp field.
    pub fn get_timestamp(&self) -> &T {
        &self.timestamp
    }

    /// Create a new [`ScObs`], with an optional observation.
    pub fn new(timestamp: T, configuration: ScObsConf<T>, opt_observation: Option<O>) -> Self
    where
        O: Default,
    {
        Self {
            configuration,
            observation: opt_observation.unwrap_or_default(),
            timestamp,
        }
    }

    /// Set the observation field.
    pub fn set_observation(&mut self, observation: O) {
        self.observation = observation;
    }
}

/// Represents a time series of spacecraft observations.
#[derive(Clone, Debug, Deserialize, IntoIterator, Serialize)]
pub struct ScObsSeries<T, O> {
    /// Vector of spacecraft observations.
    #[into_iterator(ref)]
    scobs: Vec<ScObs<T, O>>,

    /// The sorting indices that are used to recover the original [`ScObsSeries`]
    /// objects from a composite.
    sorti: Vec<usize>,
}

impl<T, O> ScObsSeries<T, O>
where
    T: PartialOrd,
{
    /// Returns the number of valid observations.
    pub fn count_observations(&self) -> usize
    where
        O: OcnusObser,
    {
        self.into_iter()
            .fold(0, |acc, next| match next.get_observation().is_valid() {
                true => acc + 1,
                false => acc,
            })
    }

    /// Returns the number of individual [`ScObsSeries`] contained within.
    ///
    /// This value corresponds to the length of the vector returned by
    /// [`ScObsSeries::sort_by_timestamp`].
    pub fn count_series(&self) -> usize {
        self.sorti.iter().fold(0, |acc, next| max(acc, *next)) + 1
    }

    /// Returns the first [`ScObs`].
    pub fn first_scobs(&self) -> Option<&ScObs<T, O>> {
        self.scobs.first()
    }

    /// Create a [`ScObsSeries`] from an iterator over `ScObs`.
    pub fn from_iterator<I: IntoIterator<Item = ScObs<T, O>>>(iter: I) -> Self {
        let scobs = iter.into_iter().collect::<Vec<ScObs<T, O>>>();
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

    /// Returns last first [`ScObs`].
    pub fn last_scobs(&self) -> Option<&ScObs<T, O>> {
        self.scobs.last()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.scobs.len()
    }

    /// Sorts the underlying `Vec<ScObs>` object by the time stamp field.
    pub fn sort_by_timestamp(&mut self) {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort_closure = |scobs: &mut Vec<ScObs<T, O>>, sorti: &mut Vec<usize>| {
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
    pub fn split(self) -> Vec<ScObsSeries<T, O>>
    where
        T: Clone,
        O: Clone,
    {
        (0..self.count_series())
            .map(|idx| {
                let scobs = zip_eq(&self.scobs, &self.sorti)
                    .filter_map(|(obs, sdx)| match sdx.cmp(&idx) {
                        Ordering::Equal => Some(obs.clone()),
                        _ => None,
                    })
                    .collect::<Vec<ScObs<T, O>>>();

                let length = scobs.len();

                Self {
                    scobs,
                    sorti: vec![0; length],
                }
            })
            .collect()
    }
}

impl<T, O> Add for ScObsSeries<T, O> {
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

impl<T, O> AddAssign for ScObsSeries<T, O> {
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
    fn test_scobs() {
        let sctc1 = ScObs {
            timestamp: 0.0,
            configuration: ScObsConf::Distance(1.0),
            observation: 0.0,
        };

        let sctc2 = ScObs {
            timestamp: 1.0,
            configuration: ScObsConf::Distance(1.0),
            observation: 0.0,
        };

        let sctc3 = ScObs {
            timestamp: 0.5,
            configuration: ScObsConf::Distance(1.0),
            observation: 0.0,
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
