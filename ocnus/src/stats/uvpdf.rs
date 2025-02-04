use crate::{
    stats::{ProbabilityDensityFunctionSampling, StatsError},
    Fp,
};
use derive_more::derive::{Deref, DerefMut, IntoIterator};
use nalgebra::SVector;
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

/// A PDF  composed of `P` independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct PUnivariatePDF<const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    [UnivariatePDF; P],
);

impl<const P: usize> PUnivariatePDF<P> {
    /// Create a new [`PUnivariatePDF`] object.
    pub fn new(uvpdfs: [UnivariatePDF; P]) -> Self {
        Self(uvpdfs)
    }
}

impl<const P: usize> ProbabilityDensityFunctionSampling<P> for &PUnivariatePDF<P> {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, P>, StatsError> {
        let mut sample = [0.0; P];

        sample
            .iter_mut()
            .zip(self.0.iter())
            .try_for_each(|(value, uvpdf)| {
                *value = uvpdf.sample(rng)?[0];

                Ok(())
            })?;

        Ok(SVector::from(sample))
    }

    fn valid_range(&self) -> [(Fp, Fp); P] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.valid_range()[0])
            .collect::<Vec<(Fp, Fp)>>()
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

impl ProbabilityDensityFunctionSampling<1> for &UnivariatePDF {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        let sample = match self {
            UnivariatePDF::Constant(pdf) => pdf.sample(rng),
            UnivariatePDF::Cosine(pdf) => pdf.sample(rng),
            UnivariatePDF::Normal(pdf) => pdf.sample(rng),
            UnivariatePDF::Reciprocal(pdf) => pdf.sample(rng),
            UnivariatePDF::Uniform(pdf) => pdf.sample(rng),
        }?;

        Ok(sample)
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
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
    constant: Fp,
}

impl ConstantPDF {
    /// Create a new [`ConstantPDF`] object.
    pub fn new(constant: Fp) -> Self {
        Self { constant }
    }

    /// Create a new [`ConstantPDF`] object wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(constant: Fp) -> UnivariatePDF {
        UnivariatePDF::Constant(Self::new(constant))
    }
}

impl ProbabilityDensityFunctionSampling<1> for &ConstantPDF {
    fn sample(&self, _rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        Ok(SVector::from([self.constant]))
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CosinePDF {
    range: (Fp, Fp),
}

impl CosinePDF {
    /// Create a new [`CosinePDF`] object.
    pub fn new(range: (Fp, Fp)) -> Result<Self, StatsError> {
        let (minv, maxv) = range;

        if (minv < -std::f64::consts::PI as Fp / 2.0)
            || (maxv > std::f64::consts::PI as Fp / 2.0)
            || (minv > maxv)
        {
            return Err(StatsError::InvalidParamRange((minv, maxv)));
        }

        Ok(Self { range })
    }

    /// Create a new [`CosinePDF`] object wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (Fp, Fp)) -> Result<UnivariatePDF, StatsError> {
        Ok(UnivariatePDF::Cosine(Self::new(range)?))
    }
}

impl ProbabilityDensityFunctionSampling<1> for &CosinePDF {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        let (minv, maxv) = self.range;

        let uniform = Uniform::new_inclusive(minv.sin(), maxv.sin()).unwrap();

        Ok(SVector::from([rng.sample(uniform).asin()]))
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NormalPDF {
    mean: Fp,
    range: (f32, f32),
    std_dev: f32,
}

impl NormalPDF {
    /// Create a new [`NormalPDF`] object.
    pub fn new(mean: Fp, std_dev: Fp, range: (f32, f32)) -> Result<Self, StatsError> {
        let (minv, maxv) = range;

        if (minv < mean) || (maxv > mean) || (minv > maxv) {
            return Err(StatsError::InvalidParamRange((minv, maxv)));
        }

        Ok(Self {
            mean,
            range,
            std_dev,
        })
    }

    /// Create a new [`NormalPDF`] object wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(
        mean: Fp,
        std_dev: Fp,
        range: (f32, f32),
    ) -> Result<UnivariatePDF, StatsError> {
        Ok(UnivariatePDF::Normal(Self::new(mean, std_dev, range)?))
    }
}

impl ProbabilityDensityFunctionSampling<1> for &NormalPDF {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        let (minv, maxv) = self.range;
        let normal = Normal::new(self.mean, self.std_dev).expect("invalid variance");

        let sample = {
            let mut candidate = rng.sample(normal);
            let mut limit_counter = 0;

            // Continsouly draw candidates until a sample is drawn within the valid range.
            while ((minv > candidate) | (candidate > maxv)) && limit_counter < 100 {
                candidate = rng.sample(normal);
                limit_counter += 1;
            }

            if limit_counter == 50 {
                return Err(StatsError::SamplerLimit(50));
            } else {
                candidate
            }
        };

        Ok(SVector::from([sample]))
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}

/// A reciprocal PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReciprocalPDF {
    range: (f32, f32),
}
impl ReciprocalPDF {
    /// Create a new [`ReciprocalPDF`] object.
    pub fn new(range: (f32, f32)) -> Result<Self, StatsError> {
        let (minv, maxv) = range;

        if (minv < 0.0) || (maxv < 0.0) || (minv > maxv) {
            return Err(StatsError::InvalidParamRange((minv, maxv)));
        }

        Ok(Self { range })
    }

    /// Create a new [`ReciprocalPDF`] object wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<UnivariatePDF, StatsError> {
        Ok(UnivariatePDF::Reciprocal(Self::new(range)?))
    }
}

impl ProbabilityDensityFunctionSampling<1> for &ReciprocalPDF {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        let (minv, maxv) = self.range;

        if (minv < 0.0) || (maxv < 0.0) {
            return Err(StatsError::InvalidParamRange((minv, maxv)));
        }

        // Inverse transform sampling.
        let ratio = maxv / minv;
        let cdf_inv = |u: f32| minv * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(0.0, 1.0).unwrap();

        Ok(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UniformPDF {
    range: (f32, f32),
}

impl UniformPDF {
    /// Create a new [`UniformPDF`] object.
    pub fn new(range: (f32, f32)) -> Result<Self, StatsError> {
        let (minv, maxv) = range;

        if minv >= maxv {
            return Err(StatsError::InvalidParamRange((minv, maxv)));
        }

        Ok(Self { range })
    }

    /// Create a new [`UniformPDF`] object wrapped within the [`UnivariatePDF`] ADT.
    pub fn new_uvpdf(range: (f32, f32)) -> Result<UnivariatePDF, StatsError> {
        Ok(UnivariatePDF::Uniform(Self::new(range)?))
    }
}

impl ProbabilityDensityFunctionSampling<1> for &UniformPDF {
    fn sample(&self, rng: &mut impl Rng) -> Result<SVector<Fp, 1>, StatsError> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        Ok(SVector::from([rng.sample(uniform)]))
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}
