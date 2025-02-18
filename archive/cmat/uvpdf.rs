use derive_more::{Deref, DerefMut, IntoIterator};
use log::warn;
use nalgebra::SVector;
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

use super::{OcnusStatsError, PDF};

/// A P-dimensional PDF composed of independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct PUnivariatePDF<const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [UnivariatePDF; P],
);

impl<const P: usize> PUnivariatePDF<P> {
    /// Create a new [`PUnivariatePDF`].
    pub fn new(uvpdfs: [UnivariatePDF; P]) -> Self {
        Self(uvpdfs)
    }
}

impl<const P: usize> PDF<P> for &PUnivariatePDF<P> {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, P>) -> f32 {
        let mut rlh = 1.0;

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.relative_density(&vec.as_view());
        });

        rlh
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, P>, OcnusStatsError> {
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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum UnivariatePDF {
    Constant(ConstantPDF),
    Cosine(CosinePDF),
    Normal(NormalPDF),
    Reciprocal(ReciprocalPDF),
    Uniform(UniformPDF),
}

impl PDF<1> for &UnivariatePDF {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        match self {
            UnivariatePDF::Constant(pdf) => pdf.relative_density(x),
            UnivariatePDF::Cosine(pdf) => pdf.relative_density(x),
            UnivariatePDF::Normal(pdf) => pdf.relative_density(x),
            UnivariatePDF::Reciprocal(pdf) => pdf.relative_density(x),
            UnivariatePDF::Uniform(pdf) => pdf.relative_density(x),
        }
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
        let sample = match self {
            UnivariatePDF::Constant(pdf) => pdf.draw_sample(rng),
            UnivariatePDF::Cosine(pdf) => pdf.draw_sample(rng),
            UnivariatePDF::Normal(pdf) => pdf.draw_sample(rng),
            UnivariatePDF::Reciprocal(pdf) => pdf.draw_sample(rng),
            UnivariatePDF::Uniform(pdf) => pdf.draw_sample(rng),
        }?;

        Ok(sample)
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        match self {
            UnivariatePDF::Constant(pdf) => pdf.valid_range(),
            UnivariatePDF::Cosine(pdf) => pdf.valid_range(),
            UnivariatePDF::Normal(pdf) => pdf.valid_range(),
            UnivariatePDF::Reciprocal(pdf) => pdf.valid_range(),
            UnivariatePDF::Uniform(pdf) => pdf.valid_range(),
        }
    }
}

/// A constant PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstantPDF {
    constant: f32,
}

impl ConstantPDF {
    /// Create a new [`ConstantPDF`].
    pub fn new(constant: f32) -> Self {
        Self { constant }
    }

    /// Create a new [`ConstantPDF`] wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(constant: f32) -> UnivariatePDF {
        UnivariatePDF::Constant(Self::new(constant))
    }
}

impl PDF<1> for &ConstantPDF {
    fn relative_density(&self, _x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        1.0
    }

    fn draw_sample(&self, _rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
        Ok(SVector::from([self.constant]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CosinePDF {
    range: (f32, f32),
}

impl CosinePDF {
    /// Create a new [`CosinePDF`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatsError> {
        let (minv, maxv) = range;

        if (minv < -std::f32::consts::PI / 2.0)
            || (maxv > std::f32::consts::PI / 2.0)
            || (minv > maxv)
        {
            return Err(OcnusStatsError::InvalidRange {
                name: "CosinePDF",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`CosinePDF`] wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<UnivariatePDF, OcnusStatsError> {
        Ok(UnivariatePDF::Cosine(Self::new(range)?))
    }
}

impl PDF<1> for &CosinePDF {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        x[0].cos()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
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
pub struct NormalPDF {
    mean: f32,
    range: (f32, f32),
    std_dev: f32,
}

impl NormalPDF {
    /// Create a new [`NormalPDF`].
    pub fn new(mean: f32, std_dev: f32, range: (f32, f32)) -> Result<Self, OcnusStatsError> {
        let (minv, maxv) = range;

        if (minv < mean) || (maxv > mean) || (minv > maxv) {
            return Err(OcnusStatsError::InvalidRange {
                name: "NormalPDF",
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

    /// Create a new [`NormalPDF`] wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(
        mean: f32,
        std_dev: f32,
        range: (f32, f32),
    ) -> Result<UnivariatePDF, OcnusStatsError> {
        Ok(UnivariatePDF::Normal(Self::new(mean, std_dev, range)?))
    }
}

impl PDF<1> for &NormalPDF {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        ((x[0] - self.mean).powi(2) / 2.0 / self.std_dev.powi(2)).exp()
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
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
                        "NormalPDF::draw_sample has failed to draw a valid sample after {} tries",
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
pub struct ReciprocalPDF {
    range: (f32, f32),
}

impl ReciprocalPDF {
    /// Create a new [`ReciprocalPDF`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatsError> {
        let (minv, maxv) = range;

        if (minv < 0.0) || (maxv < 0.0) || (minv > maxv) {
            return Err(OcnusStatsError::InvalidRange {
                name: "ReciprocalPDF",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`ReciprocalPDF`] wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<UnivariatePDF, OcnusStatsError> {
        Ok(UnivariatePDF::Reciprocal(Self::new(range)?))
    }
}

impl PDF<1> for &ReciprocalPDF {
    fn relative_density(&self, x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        let (minv, maxv) = self.range;

        1.0 / (x[0] * (maxv.ln() - minv.ln()))
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
        let (minv, maxv) = self.range;

        if (minv < 0.0) || (maxv < 0.0) {
            return Err(OcnusStatsError::InvalidRange {
                name: "ReciprocalPDF",
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
pub struct UniformPDF {
    range: (f32, f32),
}

impl UniformPDF {
    /// Create a new [`UniformPDF`].
    pub fn new(range: (f32, f32)) -> Result<Self, OcnusStatsError> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(OcnusStatsError::InvalidRange {
                name: "UniformPDF",
                maxv,
                minv,
            });
        }

        Ok(Self { range })
    }

    /// Create a new [`UniformPDF`] wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<UnivariatePDF, OcnusStatsError> {
        Ok(UnivariatePDF::Uniform(Self::new(range)?))
    }
}

impl PDF<1> for &UniformPDF {
    fn relative_density(&self, _x: &nalgebra::SVectorView<f32, 1>) -> f32 {
        1.0
    }

    fn draw_sample(&self, rng: &mut impl Rng) -> Result<SVector<f32, 1>, OcnusStatsError> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn valid_range(&self) -> [(f32, f32); 1] {
        [self.range]
    }
}
