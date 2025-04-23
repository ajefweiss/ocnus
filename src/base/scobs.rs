use derive_more::IntoIterator;
use itertools::zip_eq;
use log::debug;
use nalgebra::{DVector, RealField, Scalar, Vector3};
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Ordering, max},
    ops::{Add, AddAssign},
};

/// The configuration of a single spacecraft observation, as used in [`ScObs`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ScObsConf<T> {
    /// Position in space, in an arbitrary Solar centric coordiante system.
    Position([T; 3]),
}

/// Represents a single spacecraft observation in time, with an optional observation.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ScObs<T> {
    configuration: ScObsConf<T>,
    timestamp: T,
}

impl<T> ScObs<T> {
    /// Returns a reference to the spacecraft observation configuration.
    pub fn configuration(&self) -> &ScObsConf<T> {
        &self.configuration
    }

    /// Computes distance between `self` and `other`.
    pub fn distance(&self, other: &Self) -> T
    where
        T: Copy + RealField,
    {
        let conf_self = &self.configuration;
        let conf_other = &other.configuration;

        match (conf_self, conf_other) {
            (ScObsConf::Position(r_self), ScObsConf::Position(r_other)) => {
                (Vector3::from(*r_other) - Vector3::from(*r_self)).norm()
            }
            #[allow(unreachable_patterns)]
            _ => panic!("scobsconf types do not match"),
        }
    }

    /// Create a new [`ScObs`].
    pub fn new(timestamp: T, configuration: ScObsConf<T>) -> Self {
        Self {
            configuration,
            timestamp,
        }
    }

    /// Returns the spacecraft observation timestamp.
    pub fn timestamp(&self) -> &T {
        &self.timestamp
    }
}

/// Represents a time series of spacecraft observations.
#[derive(Clone, Debug, Default, Deserialize, IntoIterator, Serialize)]
pub struct ScObsSeries<T> {
    /// Vector of spacecraft observations.
    #[into_iterator(ref)]
    scobs: Vec<ScObs<T>>,

    /// The sorting indices that are used to recover the original [`ScObsSeries`]
    /// objects from a composite.
    sorti: Vec<usize>,
}

impl<T> ScObsSeries<T>
where
    T: PartialOrd,
{
    /// Returns the number of individual [`ScObsSeries`] contained within.
    ///
    /// This value corresponds to the length of the vector returned by
    /// [`ScObsSeries::sort_by_timestamp`].
    pub fn count_series(&self) -> usize {
        self.sorti.iter().fold(0, |acc, next| max(acc, *next)) + 1
    }

    /// Returns the first [`ScObs`].
    pub fn first_scobs(&self) -> Option<&ScObs<T>> {
        self.scobs.first()
    }

    /// Create a [`ScObsSeries`] from an iterator over `ScObs`.
    pub fn from_iterator<I: IntoIterator<Item = ScObs<T>>>(iter: I) -> Self {
        let scobs = iter.into_iter().collect::<Vec<ScObs<T>>>();
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
    pub fn last_scobs(&self) -> Option<&ScObs<T>> {
        self.scobs.last()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.scobs.len()
    }

    /// Create an empty [`ScObsSeries`]
    pub fn new() -> Self {
        Self {
            scobs: Vec::new(),
            sorti: Vec::new(),
        }
    }

    /// Sorts the underlying `Vec<ScObs>` object by the time stamp field.
    pub fn sort_by_timestamp(&mut self) {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort_closure = |scobs: &mut Vec<ScObs<T>>, sorti: &mut Vec<usize>| {
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

    /// Sorts a set of ordered data according to the order of observations.
    ///
    /// Note that this function clones the underlying observables (potentially slow).
    pub fn sort_data_by_timestamp<OT>(&self, data: &[&[OT]]) -> DVector<OT>
    where
        OT: Clone + Scalar,
    {
        let mut counters = vec![0; self.count_series()];
        let mut vector = Vec::with_capacity(self.len());

        for idx in &self.sorti {
            let d = data[*idx][counters[*idx]].clone();
            vector.push(d);

            counters[*idx] += 1;
        }

        DVector::from_iterator(vector.len(), vector.iter().cloned())
    }

    /// The reciprocal of [`ScObsSeries::sort_by_timestamp`].
    /// Calling this function consumes the [`ScObsSeries`] object and returns the original
    /// [`ScObsSeries`] objects in a vector.
    pub fn split(self) -> Vec<ScObsSeries<T>>
    where
        T: Clone,
    {
        (0..self.count_series())
            .map(|idx| {
                let scobs = zip_eq(&self.scobs, &self.sorti)
                    .filter_map(|(obs, sdx)| match sdx.cmp(&idx) {
                        Ordering::Equal => Some(obs.clone()),
                        _ => None,
                    })
                    .collect::<Vec<ScObs<T>>>();

                let length = scobs.len();

                Self {
                    scobs,
                    sorti: vec![0; length],
                }
            })
            .collect()
    }
}

impl<T> Add<ScObs<T>> for ScObsSeries<T> {
    type Output = Self;

    fn add(self, rhs: ScObs<T>) -> Self::Output {
        let mut scobs = self.scobs;
        let mut sorti = self.sorti;

        scobs.push(rhs);
        sorti.push(*sorti.last().unwrap_or(&0));

        Self { scobs, sorti }
    }
}

impl<T> Add for ScObsSeries<T> {
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

impl<T> AddAssign<ScObs<T>> for ScObsSeries<T> {
    fn add_assign(&mut self, rhs: ScObs<T>) {
        self.scobs.push(rhs);
        self.sorti.push(*self.sorti.last().unwrap_or(&0));
    }
}

impl<T> AddAssign for ScObsSeries<T> {
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

impl<T> From<&[ScObs<T>]> for ScObsSeries<T>
where
    T: Clone,
{
    fn from(value: &[ScObs<T>]) -> Self {
        Self {
            scobs: Vec::from(value),
            sorti: vec![0; value.len()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scobs() {
        let sctc1 = ScObs {
            timestamp: 0.0,
            configuration: ScObsConf::Position([1.0, 0.0, 0.0]),
        };

        let sctc2 = ScObs {
            timestamp: 1.0,
            configuration: ScObsConf::Position([1.0, 0.0, 0.0]),
        };

        let sctc3 = ScObs {
            timestamp: 0.5,
            configuration: ScObsConf::Position([1.0, 0.0, 0.0]),
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
