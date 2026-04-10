use derive_more::From;
use nalgebra::{Matrix, RealField, SMatrix, SVector, SVectorView};
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, ops::Add};

/// A newtype for storing internal coordinates and basis vectors.
#[derive(Clone, Debug, Deserialize, From, PartialEq, Serialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct ICSBasis<T, const D: usize>
where
    T: RealField,
{
    ics: SVector<T, D>,
    basis: SMatrix<T, D, D>,
}

impl<T, const D: usize> ICSBasis<T, D>
where
    T: RealField,
{
    /// Returns the basis vector corresponding to the given index.
    pub fn basis(&self) -> &SMatrix<T, D, D> {
        &self.basis
    }

    /// Returns the internal coordinates of the [`ICSBasis`] object.
    pub fn ics(&self) -> &SVector<T, D> {
        &self.ics
    }

    /// Creates a new [`ICSBasis`] object from the given internal coordinates and basis vectors.
    pub fn new(ics: SVector<T, D>, vectors: &[SVector<T, D>]) -> Self {
        Self {
            ics,
            basis: Matrix::from_columns(vectors),
        }
    }
}

impl<T, const D: usize> ICSBasis<T, D>
where
    T: RealField,
{
    /// Returns the `mu` basis vector of the [`ICSBasis`] object.
    pub fn eps_mu<'a>(&'a self) -> SVectorView<'a, T, D> {
        self.basis.column(0)
    }

    /// Returns the `nu` basis vector of the [`ICSBasis`] object.
    pub fn eps_nu<'a>(&'a self) -> SVectorView<'a, T, D> {
        self.basis.column(1)
    }

    /// Returns the `s` basis vector of the [`ICSBasis`] object.
    pub fn eps_s<'a>(&'a self) -> SVectorView<'a, T, D> {
        self.basis.column(2)
    }
}

impl<T, const D: usize> Add for ICSBasis<T, D>
where
    T: RealField,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let ics = self.ics + rhs.ics;
        let basis = self.basis + rhs.basis;

        Self { ics, basis }
    }
}

impl<T, const D: usize> Zero for ICSBasis<T, D>
where
    T: RealField,
{
    fn is_zero(&self) -> bool {
        self.ics.is_zero() && self.basis.iter().all(|e| e.is_zero())
    }

    fn set_zero(&mut self) {
        self.ics.set_zero();
        for e in &mut self.basis {
            e.set_zero();
        }
    }

    fn zero() -> Self {
        Self {
            ics: SVector::zero(),
            basis: SMatrix::zeros(),
        }
    }
}
