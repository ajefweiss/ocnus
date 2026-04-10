//! # Observation data types and noise models.

pub mod conf;
pub mod data;
mod ensbl;
pub mod noise;

use derive_more::IntoIterator;
use nalgebra::Scalar;
use serde::{Deserialize, Serialize};
use std::{
    cmp::max,
    marker::PhantomData,
    ops::{Add, AddAssign},
};

pub use ensbl::*;

use crate::obs::conf::ObsTime;

/// An individual, or composite, observer with configuration type 'OC'.
#[derive(Clone, Debug, Default, Deserialize, IntoIterator, Serialize)]
#[serde(bound(serialize = "OC: Serialize"))]
#[serde(bound(deserialize = "OC: Deserialize<'de>"))]
pub struct Obs<T, OC> {
    /// Observer configuration time-series.
    #[into_iterator(ref)]
    conf: Vec<OC>,

    /// The indices that are used to identify composite observers.
    oind: Vec<usize>,

    _phantom: PhantomData<T>,
}

impl<T, OC> Obs<T, OC>
where
    T: Scalar,
    OC: Clone,
{
    /// Combines two [`Obs`] objects into a single one.
    pub fn combine(&self, rhs: &Self) -> Self {
        let mut conf = self.conf.clone();

        conf.extend(rhs.conf.clone());

        // Calculate the maximum existing observer index within self.
        let idx_offset = self.oind.iter().fold(0, |acc: usize, &v| max(acc, v)) + 1;

        let mut oind = self.oind.clone();

        // Add index_offset to all indices in rhs.
        oind.extend(
            rhs.oind
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        Self {
            conf,
            oind,
            _phantom: PhantomData,
        }
    }

    /// Returns the observation configurations.
    pub fn configurations(&self) -> &[OC] {
        &self.conf
    }

    /// Returns the number of individual [`Obs`]'s contained within.
    pub fn count(&self) -> usize {
        self.oind.iter().fold(0, |acc, next| max(acc, *next)) + 1
    }

    /// Returns the first observation configuration, if possible.
    pub fn first(&self) -> Option<&OC> {
        self.conf.first()
    }

    /// Returns `true`` if the observation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.conf.is_empty()
    }

    /// Returns the last observation configuration, if possible.
    pub fn last(&self) -> Option<&OC> {
        self.conf.last()
    }

    /// Returns the number of elements in the observation.
    pub fn len(&self) -> usize {
        self.conf.len()
    }

    /// Create an empty [`Obs`].
    pub fn new() -> Self {
        Self {
            conf: Vec::new(),
            oind: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Uncombines the observation into a vector of individual [`Obs`] objects.
    pub fn uncombine(&self) -> Vec<Obs<T, OC>> {
        let count = self.count();
        let mut obs_vec = Vec::with_capacity(count);

        for idx in 0..count {
            let indices = self
                .oind
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v == idx { Some(i) } else { None })
                .collect::<Vec<usize>>();

            obs_vec.push(self.subset(&indices));
        }

        obs_vec
    }

    /// Returns the uncombined indices.
    pub fn uncombined_indices(&self) -> Vec<usize> {
        let count = self.count();
        let mut counts = vec![0; count];
        let mut indices = Vec::with_capacity(self.len());

        self.oind.iter().for_each(|&group| {
            indices.push(counts[group]);
            counts[group] += 1;
        });

        indices.iter_mut().enumerate().for_each(|(edx, idx)| {
            let mut group = self.oind[edx];

            while group > 0 {
                *idx += counts[group - 1];
                group -= 1;
            }
        });

        indices
    }

    /// Returns a subset of the observation based on the provided indices.
    pub fn subset(&self, indices: &[usize]) -> Self {
        let conf = indices
            .iter()
            .map(|&i| self.conf[i].clone())
            .collect::<Vec<OC>>();

        let oind = indices
            .iter()
            .map(|&i| self.oind[i])
            .collect::<Vec<usize>>();

        Self {
            conf,
            oind,
            _phantom: PhantomData,
        }
    }
}

impl<T, OC> Obs<T, OC>
where
    T: Scalar,
    OC: ObsTime<T>,
{
    /// Sorts the underlying time-series object by the observation timestamps.
    pub fn sort_by_timestamp(&mut self)
    where
        T: PartialOrd,
    {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the
        // time stamp values.
        let bubble_sort = |conf: &mut Vec<OC>, oind: &mut Vec<usize>| {
            let mut counter = 0;

            for idx in 0..(conf.len() - 1) {
                if conf[idx].timestamp() > conf[idx + 1].timestamp() {
                    conf.swap(idx, idx + 1);
                    oind.swap(idx, idx + 1);

                    counter += 1
                }
            }

            counter
        };

        let mut counter = 1;

        while counter != 0 {
            counter = bubble_sort(&mut self.conf, &mut self.oind);
        }
    }

    /// Return the timestamps for all observations.
    pub fn timestamps(&self) -> Vec<T> {
        self.conf
            .iter()
            .map(|conf| conf.timestamp())
            .collect::<Vec<T>>()
    }
}

impl<T, OC> Add for Obs<T, OC>
where
    T: Scalar,
    OC: ObsTime<T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.combine(&rhs)
    }
}

impl<T, OC> AddAssign for Obs<T, OC>
where
    T: Scalar,
    OC: ObsTime<T>,
{
    fn add_assign(&mut self, rhs: Self) {
        self.conf.extend(rhs.conf);

        // Calculate the maximum existing observer index within self.
        let idx_offset = self.oind.iter().fold(0, |acc, &v| max(acc, v)) + 1;

        // Add index_offset to all indices in rhs.
        self.oind.extend(
            rhs.oind
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );
    }
}

impl<T, OC> FromIterator<OC> for Obs<T, OC>
where
    T: Scalar,
    OC: ObsTime<T>,
{
    fn from_iter<I: IntoIterator<Item = OC>>(iter: I) -> Self {
        let conf = iter.into_iter().collect::<Vec<OC>>();
        let length = conf.len();

        Self {
            conf,
            oind: vec![0; length],
            _phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obs::conf::VecConf;
    use nalgebra::Vector3;

    #[test]
    fn test_obs() {
        let sctc1 = Obs::from_iter([VecConf::from((0.0, Vector3::new(1.0, 0.0, 0.0)))]);
        let sctc2 = Obs::from_iter([VecConf::from((1.0, Vector3::new(1.0, 0.0, 0.0)))]);
        let sctc3 = Obs::from_iter([VecConf::from((0.5, Vector3::new(1.0, 0.0, 0.0)))]);

        let sctc12 = sctc1 + sctc2;

        assert!(sctc12.count() == 2);
        assert!(sctc3.count() == 1);
        assert!(sctc3.last().unwrap().timestamp() == 0.5);

        let mut sctc123 = sctc12 + sctc3;

        assert!(sctc123.count() == 3);
        assert!(sctc123.len() == 3);
        assert!(!sctc123.is_empty());

        sctc123.sort_by_timestamp();

        assert!(sctc123.conf.first().unwrap().timestamp() == 0.0);
        assert!(sctc123.conf.last().unwrap().timestamp() == 1.0);

        let sctc_list = sctc123.uncombine();

        assert!(sctc_list.len() == 3);
        assert!(sctc_list[0].len() == 1);
        assert!(sctc_list[0].first().unwrap().timestamp() == 0.0);
        assert!(sctc_list[1].first().unwrap().timestamp() == 1.0);
        assert!(sctc_list[2].first().unwrap().timestamp() == 0.5);

        let sctc_sub = sctc123.subset(&[0, 2]);
        assert!(sctc_sub.len() == 2);
        assert!(sctc_sub.first().unwrap().timestamp() == 0.0);
        assert!(sctc_sub.last().unwrap().timestamp() == 1.0);
    }
}
