use derive_more::{Deref, DerefMut, IntoIterator};
use log::warn;
use nalgebra::{MatrixView, SVector, SVectorView, U1};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

use super::{OcnusStatisticsError, PDF};

/// A P-dimensional PDF composed of independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct PDFUnivariates<const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [PDFUnivariate; P],
);

impl<const P: usize> PDFUnivariates<P> {
    /// Create a new [`PDFUnivariates`].
    pub fn new(uvpdfs: [PDFUnivariate; P]) -> Self {
        Self(uvpdfs)
    }
}

impl<const P: usize> PDF<P> for &PDFUnivariates<P> {
    fn relative_density(&self, x: &SVectorView<f32, P>) -> f32 {
        let mut rlh = 1.0;

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.relative_density(&vec.as_view());
        });

        rlh
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, OcnusStatisticsError> {
        let mut sample = [0.0; P];

        sample
            .iter_mut()
            .zip(self.0.iter())
            .try_for_each(|(value, uvpdf)| {
                *value = uvpdf.draw_sample(rng)?[0];

                Ok(())
            })?;

        Ok(SVector::from(sample))
    }

    fn valid_range(&self) -> [(f32, f32); P] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.valid_range()[0])
            .collect::<Vec<(f32, f32)>>()
            .try_into()
            .unwrap()
    }
}

/// Algebraic data type that contains all implemented univariate PDFs.
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum PDFUnivariate {
    Constant(PDFConstant),
    Cosine(PDFCosine),
    Normal(PDFNormal),
    Reciprocal(PDFReciprocal),
    Uniform(PDFUniform),
}

impl PDF<1> for &PDFUnivariate {
    fn relative_density(&self, x: &MatrixView<f32, U1, U1>) -> f32 {
        match self {
            PDFUnivariate::Constant(pdf) => pdf.relative_density(x),
            PDFUnivariate::Cosine(pdf) => pdf.relative_density(x),
            PDFUnivariate::Normal(pdf) => pdf.relative_density(x),
            PDFUnivariate::Reciprocal(pdf) => pdf.relative_density(x),
            PDFUnivariate::Uniform(pdf) => pdf.relative_density(x),
        }
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
        let sample = match self {
            PDFUnivariate::Constant(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Cosine(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Normal(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Reciprocal(pdf) => pdf.draw_sample(rng),
            PDFUnivariate::Uniform(pdf) => pdf.draw_sample(rng),
        }?;

        Ok(sample)
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
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
pub struct PDFConstant {
    constant: f32,
}

impl PDFConstant {
    /// Create a new [`PDFConstant`].
    pub fn new(constant: f32) -> Self {
        Self { constant }
    }

    /// Create a new [`PDFConstant`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(constant: f32) -> PDFUnivariate {
        PDFUnivariate::Constant(Self::new(constant))
    }
}

impl PDF<1> for &PDFConstant {
    fn relative_density(&self, _x: &MatrixView<f32, U1, U1>) -> f32 {
        1.0
    }

    fn draw_sample(&self, _rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
        Ok(SVector::from([self.constant]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFCosine {
    range: (f32, f32),
}

impl PDFCosine {
    /// Create a new [`PDFCosine`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatisticsError> {
        let (minv, maxv) = range;

        if (minv < -std::f32::consts::PI / 2.0)
            || (maxv > std::f32::consts::PI / 2.0)
            || (minv > maxv)
        {
            return Err(OcnusStatisticsError::InvalidRange {
                name: "PDFCosine",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFCosine`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<PDFUnivariate, OcnusStatisticsError> {
        Ok(PDFUnivariate::Cosine(Self::new(range)?))
    }
}

impl PDF<1> for &PDFCosine {
    fn relative_density(&self, x: &MatrixView<f32, U1, U1>) -> f32 {
        x[0].cos()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        let (minv, maxv) = self.range;

        let uniform = Uniform::new_inclusive(minv.sin(), maxv.sin()).unwrap();

        Ok(SVector::from([rng.sample(uniform).asin()]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [self.range]
    }
}

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFNormal {
    mean: f32,
    range: (f32, f32),
    std_dev: f32,
}

impl PDFNormal {
    /// Create a new [`PDFNormal`].
    pub fn new(mean: f32, std_dev: f32, range: (f32, f32)) -> Result<Self, OcnusStatisticsError> {
        let (minv, maxv) = range;

        if (minv < mean) || (maxv > mean) || (minv > maxv) {
            return Err(OcnusStatisticsError::InvalidRange {
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
        mean: f32,
        std_dev: f32,
        range: (f32, f32),
    ) -> Result<PDFUnivariate, OcnusStatisticsError> {
        Ok(PDFUnivariate::Normal(Self::new(mean, std_dev, range)?))
    }
}

impl PDF<1> for &PDFNormal {
    fn relative_density(&self, x: &MatrixView<f32, U1, U1>) -> f32 {
        ((x[0] - self.mean).powi(2) / 2.0 / self.std_dev.powi(2)).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
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

    fn valid_range(&self) -> [(f32, f32); 1] {
        [self.range]
    }
}

/// A reciprocal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFReciprocal {
    range: (f32, f32),
}

impl PDFReciprocal {
    /// Create a new [`PDFReciprocal`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatisticsError> {
        let (minv, maxv) = range;

        if (minv < 0.0) || (maxv < 0.0) || (minv > maxv) {
            return Err(OcnusStatisticsError::InvalidRange {
                name: "PDFReciprocal",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFReciprocal`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<PDFUnivariate, OcnusStatisticsError> {
        Ok(PDFUnivariate::Reciprocal(Self::new(range)?))
    }
}

impl PDF<1> for &PDFReciprocal {
    fn relative_density(&self, x: &MatrixView<f32, U1, U1>) -> f32 {
        let (minv, maxv) = self.range;

        1.0 / (x[0] * (maxv.ln() - minv.ln()))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
        let (minv, maxv) = self.range;

        if (minv < 0.0) || (maxv < 0.0) {
            return Err(OcnusStatisticsError::InvalidRange {
                name: "PDFReciprocal",
                maxv,
                minv,
            });
        }

        // Inverse transform sampling.
        let ratio = maxv / minv;
        let cdf_inv = |u: f32| minv * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(0.0, 1.0).unwrap();

        Ok(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [self.range]
    }
}

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PDFUniform {
    range: (f32, f32),
}

impl PDFUniform {
    /// Create a new [`PDFUniform`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatisticsError> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(OcnusStatisticsError::InvalidRange {
                name: "PDFUniform",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`PDFUniform`] wrapped within the [`PDFUnivariate`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<PDFUnivariate, OcnusStatisticsError> {
        Ok(PDFUnivariate::Uniform(Self::new(range)?))
    }
}

impl PDF<1> for &PDFUniform {
    fn relative_density(&self, _x: &MatrixView<f32, U1, U1>) -> f32 {
        1.0
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatisticsError> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [self.range]
    }
}
