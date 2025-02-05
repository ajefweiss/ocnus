use crate::{Fp, OcnusObser};
use itertools::zip_eq;
use log::debug;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{max, Ordering},
    ops::{Add, AddAssign},
};

/// The configuration of a single spacecraft observation as used in [`ScObs`].
///
/// The following variants are currently implemented:
/// - [`ScConf::TimeDistance`] : Time & (x)-position of the spacecraft in a heliocentric coordinate system.
/// - [`ScConf::TimePosition`] : Time & (x,y,z)-position of the spacecraft in a heliocentric coordinate system.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum ScConf {
    /// Timestamp & Distance from the Sun.
    TimeDistance(Fp, Fp),
    /// Timestamp & Position in space in an arbitrary Solar-centric coordiante system.
    TimePosition(Fp, [Fp; 3]),
}

impl ScConf {
    pub fn get_timestamp(&self) -> Fp {
        match self {
            ScConf::TimeDistance(timestamp, ..) => *timestamp,
            ScConf::TimePosition(timestamp, ..) => *timestamp,
        }
    }
}

/// Represents a time series of spacecraft observations tied
/// to a specific generic observation type that implements [`OcnusObser`].
///
/// [`ScObs`] has, among others, two important implementations:
/// - [`ScObs::add`] : Allows composition of two [`ScObs`] objects, and sorts the underlying vectors by the time stamp value.
/// - [`ScObs::split`] : The reciprocal of one or multiple [`ScObs::add`] calls.
///   Calling this function consumes a composite [`ScObs`] object and returns the original [`ScObs`] objects in a vector.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct ScObs<O: OcnusObser> {
    /// Vector of (optional) spacecraft observations.
    obser: Vec<Option<O>>,

    /// Vector of spacecraft configurations.
    sccnf: Vec<ScConf>,

    /// The sorting indices used to recover the original [`ScObs`] objects from a composite.
    sorti: Vec<usize>,
}

impl<O: OcnusObser> ScObs<O> {
    /// Return a reference to the underlying `Vec<Option<O>>` vector as a slice.
    pub fn as_obser_slice(&self) -> &[Option<O>] {
        self.obser.as_slice()
    }

    /// Return a reference to the underlying `Vec<ScConf>` vector as a slice.
    pub fn as_scconf_slice(&self) -> &[ScConf] {
        self.sccnf.as_slice()
    }

    /// Create a new [`ScObs`] object from an iterator over `ScConf` and `Option<impl OcnusObser>`.
    /// Returns [None] if the two iterators do not match in length.
    pub fn from_iterator<I: IntoIterator<Item = (Option<O>, ScConf)>>(iter: I) -> Self {
        let (obser_list, scconf_list): (Vec<Option<O>>, Vec<ScConf>) = iter.into_iter().unzip();

        let length = obser_list.len();

        Self {
            obser: obser_list,
            sccnf: scconf_list,
            sorti: vec![0; length],
        }
    }

    /// Returns `true`` if the observation contains no elements.
    pub fn is_empty(&self) -> bool {
        self.sccnf.is_empty()
    }

    /// Returns an iterator over the observations.
    pub fn iter(&self) -> impl Iterator<Item = (&ScConf, &Option<O>)> {
        zip_eq(self.sccnf.iter(), self.obser.iter())
    }

    /// Returns an iterator that allows modifying each observation.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&mut ScConf, &mut Option<O>)> {
        zip_eq(self.sccnf.iter_mut(), self.obser.iter_mut())
    }

    /// Returns the number of elements in the observation, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.sccnf.len()
    }

    /// Returns the number of individual [`ScObs`] contained within.
    ///
    /// This value corresponds to the length of the vector returned by [`ScObs::sort_by_time`].
    pub fn nseries(&self) -> usize {
        self.sorti.iter().fold(0, |acc, value| max(acc, *value)) + 1
    }

    /// Sorts the underlying `Vec<ScTC>` object by the time stamp field.
    pub fn sort_by_time(&mut self) {
        // Implements the bubble sort algorithm for re-ordering all vectors according to the time stamp values.
        let bubble_sort_closure =
            |obser: &mut Vec<Option<O>>, sccnf: &mut Vec<ScConf>, sorti: &mut Vec<usize>| {
                let mut counter = 0;

                for idx in 0..(sccnf.len() - 1) {
                    if sccnf[idx].get_timestamp() > sccnf[idx + 1].get_timestamp() {
                        obser.swap(idx, idx + 1);
                        sccnf.swap(idx, idx + 1);
                        sorti.swap(idx, idx + 1);

                        counter += 1
                    }
                }

                counter
            };

        let mut counter = 1;

        while counter != 0 {
            counter = bubble_sort_closure(&mut self.obser, &mut self.sccnf, &mut self.sorti);
        }
    }

    /// The reciprocal of [`ScObs::sort_by_time`].
    /// Calling this function consumes the [`ScObs`] object and returns the original [`ScObs`] objects in a vector.
    pub fn split(self) -> Vec<ScObs<O>> {
        (0..self.nseries())
            .map(|idx| {
                let copy_obser = zip_eq(&self.obser, &self.sorti)
                    .filter_map(|(obs, sdx)| match sdx.cmp(&idx) {
                        Ordering::Equal => Some(obs.clone()),
                        _ => None,
                    })
                    .collect::<Vec<Option<O>>>();

                let copy_sctcl = zip_eq(&self.sccnf, &self.sorti)
                    .filter_map(|(sccnf, sdx)| match sdx.cmp(&idx) {
                        Ordering::Equal => Some(sccnf.clone()),
                        _ => None,
                    })
                    .collect::<Vec<ScConf>>();

                let length = copy_obser.len();

                ScObs {
                    obser: copy_obser,
                    sccnf: copy_sctcl,
                    sorti: vec![0; length],
                }
            })
            .collect()
    }
}

impl<O: OcnusObser> Add for ScObs<O> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new_obser = self.obser;
        let mut new_sctcl = self.sccnf;

        debug!(
            "merging two ScObs objects ({} + {})",
            new_sctcl.len(),
            rhs.sccnf.len()
        );

        new_obser.extend(rhs.obser);
        new_sctcl.extend(rhs.sccnf);

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |maxv, &v| max(maxv, v)) + 1;

        let mut new_sorti = self.sorti;

        // Add index_offset to all indices in rhs.
        new_sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        let mut result = Self {
            obser: new_obser,
            sccnf: new_sctcl,
            sorti: new_sorti,
        };

        result.sort_by_time();

        result
    }
}

impl<O: OcnusObser> AddAssign for ScObs<O> {
    fn add_assign(&mut self, rhs: Self) {
        debug!(
            "merging two ScObs objects ({} + {})",
            self.sccnf.len(),
            rhs.sccnf.len()
        );

        self.obser.extend(rhs.obser);
        self.sccnf.extend(rhs.sccnf);

        // Calculate the maximum existing spacecraft index within self.
        let idx_offset = self.sorti.iter().fold(0, |maxv, &v| max(maxv, v)) + 1;

        // Add index_offset to all indices in rhs.
        self.sorti.extend(
            rhs.sorti
                .iter()
                .map(|sdx| sdx + idx_offset)
                .collect::<Vec<usize>>(),
        );

        self.sort_by_time();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scobs_slice() {
        let sctc1 = (None, ScConf::TimeDistance(0.0, 1.0));
        let sctc2 = (None, ScConf::TimeDistance(1.0, 1.0));
        let sctc3 = (None, ScConf::TimeDistance(0.5, 1.0));

        let mut ts3: ScObs<Fp> = ScObs::<Fp>::from_iterator([sctc1, sctc2]);
        let ts4: ScObs<Fp> = ScObs::<Fp>::from_iterator([sctc3]);

        assert!(ts3.nseries() == 1);
        assert!(ts4.nseries() == 1);
        assert!(ts4.iter().last().unwrap().0.get_timestamp() == 0.5);

        ts3 += ts4.clone();

        assert!(ts3.nseries() == 2);

        assert!(ts3.sccnf.first().unwrap().get_timestamp() == 0.0);
        assert!(ts3.sccnf.last().unwrap().get_timestamp() == 1.0);

        let unsort = ts3.clone().split();

        assert!(unsort.len() == 2);
        assert!(unsort[0].sccnf.first().unwrap().get_timestamp() == 0.0);
        assert!(unsort[0].sccnf.last().unwrap().get_timestamp() == 1.0);

        assert!(unsort[1].sccnf.first().unwrap().get_timestamp() == 0.5);

        let ts5 = ts3 + ts4;

        assert!(ts5.sccnf.first().unwrap().get_timestamp() == 0.0);
        assert!(ts5.sccnf.last().unwrap().get_timestamp() == 1.0);
    }
}
