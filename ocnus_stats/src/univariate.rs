use crate::{OcnusPDF, StatsError};
use derive_more::{Deref, DerefMut, IntoIterator};
use log::error;
use nalgebra::{MatrixView, RealField, SVector, SVectorView, Scalar, U1};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, usize};

/// A joint probability density function for `N` independent variables.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct UnivariateND<T, const N: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    #[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
    [Univariate1D<T>; N],
);

impl<T, const N: usize> UnivariateND<T, N> {
    /// Create a new [`UnivariateND`].
    pub fn new(uvpdfs: [Univariate1D<T>; N]) -> Self {
        Self(uvpdfs)
    }
}

impl<T, const N: usize> OcnusPDF<T, N> for &UnivariateND<T, N>
where
    T: Copy + RealField + SampleUniform + Scalar,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &SVectorView<T, N>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let mut rlh = T::one();

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.density_rel(&vec.as_view());
        });

        rlh
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, N>, StatsError<T>> {
        let mut sample = [T::zero(); N];

        sample
            .iter_mut()
            .zip(self.0.iter())
            .try_for_each(|(value, uvpdf)| {
                *value = uvpdf.draw_sample(rng)?[0];

                Ok(())
            })?;

        Ok(SVector::from(sample))
    }

    fn get_valid_range(&self) -> [(T, T); N] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.get_valid_range()[0])
            .collect::<Vec<(T, T)>>()
            .try_into()
            .unwrap()
    }
}

/// The algebraic data type for representing univariate probability density functions.
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum Univariate1D<T> {
    Constant(Constant1D<T>),
    Cosine(Cosine1D<T>),
    Normal(Normal1D<T>),
    Reciprocal(Reciprocal1D<T>),
    Uniform(Uniform1D<T>),
}

impl<T> OcnusPDF<T, 1> for &Univariate1D<T>
where
    T: Copy + RealField + SampleUniform + Scalar,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        match self {
            Univariate1D::Constant(pdf) => pdf.density_rel(x),
            Univariate1D::Cosine(pdf) => pdf.density_rel(x),
            Univariate1D::Normal(pdf) => pdf.density_rel(x),
            Univariate1D::Reciprocal(pdf) => pdf.density_rel(x),
            Univariate1D::Uniform(pdf) => pdf.density_rel(x),
        }
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let sample = match self {
            Univariate1D::Constant(pdf) => pdf.draw_sample(rng),
            Univariate1D::Cosine(pdf) => pdf.draw_sample(rng),
            Univariate1D::Normal(pdf) => pdf.draw_sample(rng),
            Univariate1D::Reciprocal(pdf) => pdf.draw_sample(rng),
            Univariate1D::Uniform(pdf) => pdf.draw_sample(rng),
        }?;

        Ok(sample)
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        match self {
            Univariate1D::Constant(pdf) => pdf.get_valid_range(),
            Univariate1D::Cosine(pdf) => pdf.get_valid_range(),
            Univariate1D::Normal(pdf) => pdf.get_valid_range(),
            Univariate1D::Reciprocal(pdf) => pdf.get_valid_range(),
            Univariate1D::Uniform(pdf) => pdf.get_valid_range(),
        }
    }
}

/// A constant probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Constant1D<T> {
    constant: T,
}

impl<T> Constant1D<T> {
    /// Create a new [`Univariate1D`].
    pub fn new(constant: T) -> Univariate1D<T> {
        Univariate1D::Constant(Self { constant })
    }
}

impl<T> OcnusPDF<T, 1> for &Constant1D<T>
where
    T: Copy + RealField + Scalar,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }

    fn draw_sample(&self, _rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        Ok(SVector::from([self.constant]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Cosine1D<T> {
    range: (T, T),
}

impl<T> Cosine1D<T>
where
    T: Copy + RealField,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv < -T::frac_pi_2()) || (maxv > T::frac_pi_2()) || (minv > maxv) {
            return Err(StatsError::InvalidRange {
                name: "Cosine1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Cosine(Self { range }))
    }
}

impl<T> OcnusPDF<T, 1> for &Cosine1D<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        x[0].cos()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        let (minv, maxv) = self.range;

        let uniform = Uniform::new_inclusive(minv.sin(), maxv.sin()).unwrap();

        Ok(SVector::from([rng.sample(uniform).asin()]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A univariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Normal1D<T> {
    mean: T,
    range: (T, T),
    std_dev: T,
}

impl<T> Normal1D<T>
where
    T: Copy + RealField,
{
    /// Create a new [`Univariate1D`].
    pub fn new(mean: T, std_dev: T, range: (T, T)) -> Result<Univariate1D<T>, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv > mean) || (maxv < mean) || (minv > maxv) {
            return Err(StatsError::InvalidRange {
                name: "Normal1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Normal(Self {
            mean,
            std_dev,
            range,
        }))
    }
}

impl<T> OcnusPDF<T, 1> for &Normal1D<T>
where
    T: Copy + RealField + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        ((x[0] - self.mean) / self.std_dev).powi(2) / T::from_usize(2).unwrap()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;
        let normal = StandardNormal;

        let sample = {
            let mut attempts = 0;
            let mut candidate = self.std_dev * rng.sample(normal) + self.mean;

            // Continsouly draw candidates until a sample is drawn within the valid range.
            while (minv > candidate) | (candidate > maxv) {
                candidate = rng.sample(normal);

                attempts += 1;

                if attempts > 99 {
                    error!(
                        "Normal1D::draw_sample has failed to draw a valid sample after {} tries",
                        attempts
                    );

                    return Err(StatsError::InefficientSampling {
                        name: "Normal1D",
                        count: 100,
                    });
                }
            }

            candidate
        };

        Ok(SVector::from([sample]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A reciprocal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Reciprocal1D<T> {
    range: (T, T),
}

impl<T> Reciprocal1D<T>
where
    T: Copy + RealField + SampleUniform,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv <= T::zero()) || (maxv < T::zero()) || (minv > maxv) {
            return Err(StatsError::InvalidRange {
                name: "Reciprocal1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Reciprocal(Self { range }))
    }
}

impl<T> OcnusPDF<T, 1> for &Reciprocal1D<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let (minv, maxv) = self.range;

        T::one() / (x[0] * (maxv.ln() - minv.ln()))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;

        if (minv < T::zero()) || (maxv < T::zero()) {
            return Err(StatsError::InvalidRange {
                name: "Reciprocal1D",
                maxv,
                minv,
            });
        }

        // Inverse transform sampling.
        let ratio = maxv / minv;
        let cdf_inv = |u: T| minv * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(T::zero(), T::one()).unwrap();

        Ok(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A uniform probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Uniform1D<T> {
    range: (T, T),
}

impl<T> Uniform1D<T>
where
    T: Copy + RealField,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, StatsError<T>> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(StatsError::InvalidRange {
                name: "Uniform1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Uniform(Self { range }))
    }
}

impl<T> OcnusPDF<T, 1> for &Uniform1D<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_univariatend() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let uvpdf = UnivariateND::new([
            Constant1D::new(1.0),
            Cosine1D::new((0.1, 0.2)).unwrap(),
            Normal1D::new(0.1, 0.25, (-0.5, 1.5)).unwrap(),
            Reciprocal1D::new((0.1, 0.5)).unwrap(),
            Uniform1D::new((1.0, 2.0)).unwrap(),
        ]);

        assert!(
            ((&uvpdf).density_rel(&SVector::from([1.0, 0.15, 0.15, 0.2, 1.5]).as_view())
                - 0.06143580130038274f32)
                .abs()
                < 1e-6
        );

        assert!(
            (&uvpdf)
                .density_rel(&SVector::from([1.0, 0.05, 0.15, 0.2, 1.5]).as_view())
                .is_nan()
        );

        assert!(
            ((&uvpdf).draw_sample(&mut rng).unwrap()
                - SVector::from([
                    1.0,
                    0.1810371254631513,
                    0.2788901781826483,
                    0.11749042572818454,
                    1.7462168706168106,
                ]))
            .norm()
                < 1e-6
        );

        assert!((&uvpdf).validate_sample(&(&uvpdf).draw_sample(&mut rng).unwrap().as_view()));
    }
}
