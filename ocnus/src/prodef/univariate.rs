use crate::{
    fXX,
    math::{T, asin, cos, exp, ln, powi, sin},
    prodef::{OcnusProDeF, ProDeFError},
};
use derive_more::{Deref, DerefMut, IntoIterator};
use log::warn;
use nalgebra::{MatrixView, SVector, SVectorView, U1};
use rand::Rng;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A P-dimensional PDF composed of P independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct UnivariateND<T, const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    #[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
    [Univariate1D<T>; P],
);

impl<T, const P: usize> UnivariateND<T, P> {
    /// Create a new [`UnivariateND`].
    pub fn new(uvpdfs: [Univariate1D<T>; P]) -> Self {
        Self(uvpdfs)
    }
}

impl<T, const P: usize> OcnusProDeF<T, P> for &UnivariateND<T, P>
where
    T: fXX + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &SVectorView<T, P>) -> T {
        let mut rlh = T::one();

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.density_rel(&vec.as_view());
        });

        rlh
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, P>, ProDeFError<T>> {
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

    fn get_valid_range(&self) -> [(T, T); P] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.get_valid_range()[0])
            .collect::<Vec<(T, T)>>()
            .try_into()
            .unwrap()
    }
}

/// Algebraic data type that contains all implemented single dimensional univariate PDFs.
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

impl<T> OcnusProDeF<T, 1> for &Univariate1D<T>
where
    T: fXX + SampleUniform,
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

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
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

/// A constant PDF.
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

impl<T> OcnusProDeF<T, 1> for &Constant1D<T>
where
    T: fXX,
{
    fn density_rel(&self, _x: &MatrixView<T, U1, U1>) -> T {
        T::one()
    }

    fn draw_sample(&self, _rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
        Ok(SVector::from([self.constant]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Cosine1D<T> {
    range: (T, T),
}

impl<T> Cosine1D<T>
where
    T: fXX,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, ProDeFError<T>> {
        let (minv, maxv) = range;

        if (minv < -T::half_pi()) || (maxv > T::half_pi()) || (minv > maxv) {
            return Err(ProDeFError::InvalidRange {
                name: "Cosine1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Cosine(Self { range }))
    }
}

impl<T> OcnusProDeF<T, 1> for &Cosine1D<T>
where
    T: fXX + SampleUniform,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        cos!(x[0])
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        let (minv, maxv) = self.range;

        let uniform = Uniform::new_inclusive(sin!(minv), sin!(maxv)).unwrap();

        Ok(SVector::from([asin!(rng.sample(uniform))]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Normal1D<T> {
    mean: T,
    range: (T, T),
    std_dev: T,
}

impl<T> Normal1D<T>
where
    T: fXX,
{
    /// Create a new [`Univariate1D`].
    pub fn new(mean: T, std_dev: T, range: (T, T)) -> Result<Univariate1D<T>, ProDeFError<T>> {
        let (minv, maxv) = range;

        if (minv > mean) || (maxv < mean) || (minv > maxv) {
            return Err(ProDeFError::InvalidRange {
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

impl<T> OcnusProDeF<T, 1> for &Normal1D<T>
where
    T: fXX,
    StandardNormal: Distribution<T>,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        exp!(powi!(x[0] - self.mean, 2) / T!(2.0) / powi!(self.std_dev, 2))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
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
                        "Normal1D::draw_sample has failed to draw a valid sample after {} tries",
                        attempts
                    );
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

/// A reciprocal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Reciprocal1D<T> {
    range: (T, T),
}

impl<T> Reciprocal1D<T>
where
    T: fXX + SampleUniform,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, ProDeFError<T>> {
        let (minv, maxv) = range;

        if (minv <= T::zero()) || (maxv < T::zero()) || (minv > maxv) {
            return Err(ProDeFError::InvalidRange {
                name: "Reciprocal1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Reciprocal(Self { range }))
    }
}

impl<T> OcnusProDeF<T, 1> for &Reciprocal1D<T>
where
    T: fXX + SampleUniform,
{
    fn density_rel(&self, x: &MatrixView<T, U1, U1>) -> T {
        let (minv, maxv) = self.range;

        T::one() / (x[0] * (ln!(maxv) - ln!(minv)))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
        let (minv, maxv) = self.range;

        if (minv < T::zero()) || (maxv < T::zero()) {
            return Err(ProDeFError::InvalidRange {
                name: "Reciprocal1D",
                maxv,
                minv,
            });
        }

        // Inverse transform sampling.
        let ratio = maxv / minv;
        let cdf_inv = |u: T| minv * exp!(ln!(ratio) * u);
        let uniform = Uniform::new_inclusive(T::zero(), T::one()).unwrap();

        Ok(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Uniform1D<T> {
    range: (T, T),
}

impl<T> Uniform1D<T>
where
    T: fXX + SampleUniform,
{
    /// Create a new [`Univariate1D`].
    pub fn new(range: (T, T)) -> Result<Univariate1D<T>, ProDeFError<T>> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(ProDeFError::InvalidRange {
                name: "Uniform1D",
                maxv,
                minv,
            });
        }

        Ok(Univariate1D::Uniform(Self { range }))
    }
}

impl<T> OcnusProDeF<T, 1> for &Uniform1D<T>
where
    T: fXX + SampleUniform,
{
    fn density_rel(&self, _x: &MatrixView<T, U1, U1>) -> T {
        T::one()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<T, 1>, ProDeFError<T>> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn get_valid_range(&self) -> [(T, T); 1] {
        [self.range]
    }
}
