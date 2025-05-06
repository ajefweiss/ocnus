use nalgebra::{Const, Dyn, Matrix, RealField, VecStorage};
use serde::{Deserialize, Serialize};
use std::io::Write;

/// A data structure that stores an ensemble of model parameters and states.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OcnusEnsbl<T, const P: usize, FMST, CSST>
where
    T: RealField,
{
    /// Ensemble model parameters.
    pub params_array: Matrix<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>,

    /// Forward model states
    pub fm_states: Vec<FMST>,

    /// Coordinate system states.
    pub cs_states: Vec<CSST>,

    /// Ensemble weights.
    pub weights: Vec<T>,
}

impl<T, const P: usize, FMST, CSST> OcnusEnsbl<T, P, FMST, CSST>
where
    T: RealField,
{
    /// Returns true if the ensemble contains no members.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of members in the ensemble, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.params_array.ncols()
    }

    /// Create a new [`OcnusEnsbl`] filled with zeros.
    pub fn new(size: usize) -> Self
    where
        FMST: Clone + Default,
        CSST: Clone + Default,
    {
        Self {
            params_array: Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::zeros(size),
            fm_states: vec![FMST::default(); size],
            cs_states: vec![CSST::default(); size],
            weights: vec![T::one() / T::from_usize(size).unwrap(); size],
        }
    }

    /// Serialize ensemble to a JSON file.
    pub fn save(&self, path: String) -> std::io::Result<()>
    where
        Self: Serialize,
    {
        let mut file = std::fs::File::create(path)?;

        file.write_all(serde_json5::to_string(&self).unwrap().as_bytes())?;

        Ok(())
    }
}
