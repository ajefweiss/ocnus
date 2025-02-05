use crate::OcnusObser;
use derive_more::{derive::DerefMut, Deref, From, Index, IndexMut, IntoIterator};
use itertools::zip_eq;
use ndarray::Array2;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    ops::{Add, Mul, Sub},
};

/// Generic N-dimensional observation array.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, Serialize)]
pub struct ModelObserArray<const N: usize>(pub Array2<Option<ModelObserVec<N>>>);

impl<const N: usize> ModelObserArray<N> {}

/// Generic N-dimensional observation vector.
#[derive(Clone, Debug, Deref, Deserialize, From, Index, IndexMut, IntoIterator, Serialize)]
pub struct ModelObserVec<const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    pub [f32; N],
);

impl<const N: usize> OcnusObser for ModelObserVec<N> {}

impl<const N: usize> ModelObserVec<N> {
    pub fn mse(&self, other: &Self) -> f32 {
        (self - other).sum_of_squares() / N as f32
    }

    pub fn rmse(&self, other: &Self) -> f32 {
        ((self - other).sum_of_squares() / N as f32).sqrt()
    }
}

impl<const N: usize> ModelObserVec<N> {
    /// Sum of the squares of for the underlying vector.
    pub fn sum_of_squares(&self) -> f32 {
        self.iter().map(|value| value.powi(2)).sum::<f32>()
    }

    /// Return a observation vector filled with zeros.
    pub fn zeros() -> Self {
        ModelObserVec([0.0; N])
    }
}

impl<const N: usize> Default for ModelObserVec<N> {
    fn default() -> Self {
        ModelObserVec([f32::nan(); N])
    }
}

// TODO: Generic Implementation!
impl Display for ModelObserVec<3> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.2} | {:.2} | {:.2}]", self[0], self[1], self[2])
    }
}

impl Display for ModelObserVec<4> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.2} | {:.2} | {:.2} | {:.2}]",
            self[0], self[1], self[2], self[3]
        )
    }
}

// Implement addition, subtraction and commutative multiplication for both ModelObserVec< N> and its references.
impl<const N: usize> Add for ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn add(self, rhs: Self) -> Self::Output {
        ModelObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(a, b)| a + b)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Add<&'a ModelObserVec<N>> for &'a ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn add(self, rhs: &'a ModelObserVec<N>) -> Self::Output {
        ModelObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 + v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<f32> for ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        ModelObserVec::<N>(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<f32> for &ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn mul(self, rhs: f32) -> Self::Output {
        ModelObserVec::<N>(
            self.iter()
                .map(|v| *v * rhs)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Mul<ModelObserVec<N>> for f32 {
    type Output = ModelObserVec<N>;

    fn mul(self, rhs: ModelObserVec<N>) -> Self::Output {
        ModelObserVec::<N>(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Mul<&'a ModelObserVec<N>> for f32 {
    type Output = ModelObserVec<N>;

    fn mul(self, rhs: &'a ModelObserVec<N>) -> Self::Output {
        ModelObserVec::<N>(
            rhs.iter()
                .map(|v| *v * self)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Sub for ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        ModelObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| v1 - v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<'a, const N: usize> Sub<&'a ModelObserVec<N>> for &'a ModelObserVec<N> {
    type Output = ModelObserVec<N>;

    fn sub(self, rhs: &'a ModelObserVec<N>) -> Self::Output {
        ModelObserVec::<N>(
            zip_eq(self, rhs)
                .map(|(v1, v2)| *v1 - *v2)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obsvec() {
        let ov_d = ModelObserVec::<3>::default();

        assert!(ov_d.0[0].is_nan());
        assert!(ov_d.0[1].is_nan());
        assert!(ov_d.0[2].is_nan());

        let ov_1 = ModelObserVec::<3>::from([1.0, 2.0, 3.0]);

        assert!(ov_1.sum_of_squares() == 14.0);

        let ov_2 = ModelObserVec::<3>::from([2.0, 3.0, 1.0]);

        assert!((ov_1.clone() + ov_2.clone()).sum_of_squares() == 9.0 + 25.0 + 16.0);
        assert!((ov_1.clone() - ov_2.clone()).sum_of_squares() == 6.0);

        assert!((&ov_1 + &ov_2).sum_of_squares() == 9.0 + 25.0 + 16.0);
        assert!((&ov_1 - &ov_2).sum_of_squares() == 6.0);

        assert!((ov_1.clone() * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * ov_1.clone()).sum_of_squares() == 4.0 * 14.0);

        assert!((&ov_1 * 2.0).sum_of_squares() == 4.0 * 14.0);
        assert!((2.0 * &ov_1).sum_of_squares() == 4.0 * 14.0);
    }
}
