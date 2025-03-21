use std::fmt::Debug;

use crate::stats::{PDF, StatsError};
use derive_more::{Deref, DerefMut, IntoIterator};
use log::warn;
use nalgebra::{MatrixView, RealField, SVector, SVectorView, U1};
use num_traits::{Float, FromPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A P-dimensional PDF composed of independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct PDFUnivariates<T, const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [PDFUnivariate<T>; P],
)
where
    T: for<'x> Deserialize<'x> + Serialize;

impl<T, const P: usize> PDFUnivariates<T, P>
where
    T: for<'x> Deserialize<'x> + Serialize,
{
    /// Create a new [`PDFUnivariates`].
    pub fn new(uvpdfs: [PDFUnivariate<T>; P]) -> Self {
        Self(uvpdfs)
    }
}

impl<T, const P: usize> PDF<T, P> for &PDFUnivariates<T, P>
where
    T: Copy + for<'x> Deserialize<'x> + Float + PartialOrd + RealField + SampleUniform + Serialize,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, x: &SVectorView<T, P>) -> T {
        let mut rlh = T::one();

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.relative_density(&vec.as_view());
        });

        rlh
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, StatsError<T>> {
        let mut sample = [T::zero(); P];

        sample
            .iter_mut()
            .zip(self.0.iter())
            .try_for_each(|(value, uvpdf)| {
                *value = uvpdf.draw_sample(rng)?[0];

                Ok(())
            })?;

        Ok(SVector::from(sample))
    }

    fn valid_range(&self) -> [(T, T); P] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.valid_range()[0])
            .collect::<Vec<(T, T)>>()
            .try_into()
            .unwrap()
    }
}

/// Algebraic data type that contains all implemented univariate PDFs.
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum PDFUnivariate<T> {
    Constant(PDFConstant<T>),
    Cosine(PDFCosine<T>),
    Normal(PDFNormal<T>),
    Reciprocal(PDFReciprocal<T>),
    Uniform(PDFUniform<T>),
}

impl<T> PDF<T, 1> for &PDFUnivariate<T>
where
    T: Float + RealField + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        match self {
            PDFUnivariate::Constant(pdf) => pdf.relative_density(x),
            PDFUnivariate::Cosine(pdf) => pdf.relative_density(x),
            PDFUnivariate::Normal(pdf) => pdf.relative_density(x),
            PDFUnivariate::Reciprocal(pdf) => pdf.relative_density(x),
            PDFUnivariate::Uniform(pdf) => pdf.relative_density(x),
        }
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let sample = match self {
            PDFUnivariate::Constant(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Cosine(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Normal(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Reciprocal(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Uniform(pdf) => pdf.draw_sample(rng),
        }?;

        Ok(sample)
    }

    fn valid_range(&self) -> [(T, T); 1] {
        match self {
            PDFUnivariate::Constant(pdf) => pdf.valid_range(),
            PDFUnivariate::Cosine(pdf) => pdf.valid_range(),
            PDFUnivariate::Normal(pdf) => pdf.valid_range(),
            PDFUnivariate::Reciprocal(pdf) => pdf.valid_range(),
            PDFUnivariate::Uniform(pdf) => pdf.valid_range(),
        }
    }
}

/// A constant PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFConstant<T> {
    constant: T,
}

impl<T> PDFConstant<T> {
    /// Create a new [`PDFConstant`].
    pub fn new(constant: T) -> Self {
        Self { constant }
    }

    /// Create a new [`PDFConstant`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(constant: T) -> PDFUnivariate<T> {
        PDFUnivariate::Constant(Self::new(constant))
    }
}

impl<T> PDF<T, 1> for &PDFConstant<T>
where
    T: Copy + Debug + PartialOrd + RealField + Sync,
{
    fn relative_density(&self, _x: &MatrixView<T, U1, U1>) -> T {
        T::one()
    }

    fn draw_sample(&self, _rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        Ok(SVector::from([self.constant]))
    }

    fn valid_range(&self) -> [(T, T); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFCosine<T> {
    range: (T, T),
}

impl<T> PDFCosine<T>
where
    T: Copy + RealField + PartialOrd,
{
    /// Create a new [`PDFCosine`].
    pub fn new(range: (T, T)) -> Result<Self, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv < -T::pi() / T::from_f64(2.0).unwrap())
            || (maxv > T::pi() / T::from_f64(2.0).unwrap())
            || (minv > maxv)
        {
            return Err(StatsError::InvalidRange {
                name: "PDFCosine",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFCosine`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (T, T)) -> Result<PDFUnivariate<T>, StatsError<T>> {
        Ok(PDFUnivariate::Cosine(Self::new(range)?))
    }
}

impl<T> PDF<T, 1> for &PDFCosine<T>
where
    T: Copy + Float + RealField + SampleUniform,
{
    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        Float::cos(x[0])
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        let (minv, maxv) = self.range;

        let uniform = Uniform::new_inclusive(Float::sin(minv), Float::sin(maxv)).unwrap();

        Ok(SVector::from([Float::asin(rng.sample(uniform))]))
    }

    fn valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFNormal<T> {
    mean: T,
    range: (T, T),
    std_dev: T,
}

impl<T> PDFNormal<T>
where
    T: Copy + PartialOrd,
{
    /// Create a new [`PDFNormal`].
    pub fn new(mean: T, std_dev: T, range: (T, T)) -> Result<Self, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv > mean) || (maxv < mean) || (minv > maxv) {
            return Err(StatsError::InvalidRange {
                name: "PDFNormal",
                maxv,
                minv,
            });
        }

        Ok(Self {
            mean,
            range,
            std_dev,
        })
    }

    /// Create a new [`PDFNormal`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(
        mean: T,
        std_dev: T,
        range: (T, T),
    ) -> Result<PDFUnivariate<T>, StatsError<T>> {
        Ok(PDFUnivariate::Normal(Self::new(mean, std_dev, range)?))
    }
}

impl<T> PDF<T, 1> for &PDFNormal<T>
where
    T: Copy + Float + FromPrimitive + PartialOrd + RealField,
    StandardNormal: Distribution<T>,
{
    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        Float::exp(
            Float::powi(x[0] - self.mean, 2)
                / T::from_f64(2.0).unwrap()
                / Float::powi(self.std_dev, 2),
        )
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;
        let normal = Normal::new(self.mean, self.std_dev).expect("invalid variance");

        let sample = {
            let mut attempts = 0;
            let mut candidate = rng.sample(normal);

            // Continsouly draw candidates until a sample is drawn within the valid range.
            while (minv > candidate) | (candidate > maxv) {
                candidate = rng.sample(normal);

                attempts += 1;

                if (attempts > 50) && (attempts % 50 == 0) {
                    warn!(
                        "PDFNormal::draw_sample has failed to draw a valid sample after {} tries",
                        attempts
                    );
                }
            }

            candidate
        };

        Ok(SVector::from([sample]))
    }

    fn valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A reciprocal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFReciprocal<T> {
    range: (T, T),
}

impl<T> PDFReciprocal<T>
where
    T: Copy + RealField + SampleUniform,
{
    /// Create a new [`PDFReciprocal`].
    pub fn new(range: (T, T)) -> Result<Self, StatsError<T>> {
        let (minv, maxv) = range;

        if (minv <= T::zero()) || (maxv < T::zero()) || (minv > maxv) {
            return Err(StatsError::InvalidRange {
                name: "PDFReciprocal",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFReciprocal`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (T, T)) -> Result<PDFUnivariate<T>, StatsError<T>> {
        Ok(PDFUnivariate::Reciprocal(Self::new(range)?))
    }
}

impl<T> PDF<T, 1> for &PDFReciprocal<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        let (minv, maxv) = self.range;

        T::one() / (x[0] * (maxv.ln() - minv.ln()))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;

        if (minv < T::zero()) || (maxv < T::zero()) {
            return Err(StatsError::InvalidRange {
                name: "PDFReciprocal",
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

    fn valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFUniform<T> {
    range: (T, T),
}

impl<T> PDFUniform<T>
where
    T: Copy + PartialOrd + RealField + SampleUniform,
{
    /// Create a new [`PDFUniform`].
    pub fn new(range: (T, T)) -> Result<Self, StatsError<T>> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(StatsError::InvalidRange {
                name: "PDFUniform",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFUniform`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (T, T)) -> Result<PDFUnivariate<T>, StatsError<T>> {
        Ok(PDFUnivariate::Uniform(Self::new(range)?))
    }
}

impl<T> PDF<T, 1> for &PDFUniform<T>
where
    T: Copy + PartialOrd + RealField + SampleUniform,
{
    fn relative_density(&self, _x: &MatrixView<T, U1, U1>) -> T {
        T::one()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, StatsError<T>> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}
