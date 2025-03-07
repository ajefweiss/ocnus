use derive_builder::Builder;
use derive_more::derive::Deref;
use itertools::zip_eq;
use log::debug;
use log::info;
use nalgebra::{Const, DVector, DimAdd, Dyn, SVector, ToTypenum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, mem::replace, time::Instant};

use crate::stats::PDFParticles;

use super::FEVMDataPairs;

/// A data structure that holds particle filter data and settings.
#[derive(Builder, Debug, Default, Deref, Deserialize, Serialize)]
pub struct ParticleFilterSettings<'a, FS, GS, const P: usize, const N: usize> {
    /// The underlying particle density that is filtered.
    #[deref]
    pub density: PDFParticles<P>,

    /// FEVM data pairs.
    pub fevmdp: FEVMDataPairs<P, N, FS, GS>,
}
