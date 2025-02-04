use crate::{
    alias::PMatrixViewMut,
    stats::{ProbabilityDensityFunctionSampling, StatsError},
    Fp,
};
use derive_more::derive::{Deref, DerefMut, IntoIterator};
use nalgebra::{Const, Dim, Dyn, U1};
use rand::Rng;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

/// A probability density function (PDF) composed of `P` independent univariate PDFs.
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct UnivariatePDFs<const P: usize>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    pub [UnivariatePDF; P],
);

impl<const P: usize> ProbabilityDensityFunctionSampling<P> for &UnivariatePDFs<P> {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<Const<P>, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        self.0
            .iter()
            .zip(pmatrix.row_iter_mut())
            .try_for_each(|(uvpdf, mut col)| uvpdf.sample_fill(&mut col, rng))?;

        Ok(())
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

/// Algebraic data type that contains all implemented univariate probability density functions.
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
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        match self {
            UnivariatePDF::Constant(pdf) => pdf.sample_fill(pmatrix, rng),
            UnivariatePDF::Cosine(pdf) => pdf.sample_fill(pmatrix, rng),
            UnivariatePDF::Normal(pdf) => pdf.sample_fill(pmatrix, rng),
            UnivariatePDF::Reciprocal(pdf) => pdf.sample_fill(pmatrix, rng),
            UnivariatePDF::Uniform(pdf) => pdf.sample_fill(pmatrix, rng),
        }?;

        Ok(())
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

/// A constant probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstantPDF {
    pub constant: Fp,
}

impl ProbabilityDensityFunctionSampling<1> for &ConstantPDF {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        _rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        pmatrix.iter_mut().for_each(|col| *col = self.constant);

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [(self.constant, self.constant)]
    }
}

/// A cosine normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CosinePDF {
    pub range: Option<(Fp, Fp)>,
}

impl ProbabilityDensityFunctionSampling<1> for &CosinePDF {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        // The range is limited to the interval [-π/2, π/2].
        let (minv, maxv) = match self.range {
            Some(range) => range,
            None => (
                -std::f64::consts::PI as Fp / 2.0,
                std::f64::consts::PI as Fp / 2.0,
            ),
        };

        let uniform = Uniform::new_inclusive(minv.sin(), maxv.sin()).unwrap();

        pmatrix
            .iter_mut()
            .for_each(|col| *col = rng.sample(uniform).asin());

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range.unwrap_or((
            -std::f64::consts::PI as Fp / 2.0,
            std::f64::consts::PI as Fp / 2.0,
        ))]
    }
}

/// A univariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NormalPDF {
    pub mean: Fp,
    pub range: Option<(f32, f32)>,
    pub std_dev: f32,
}

impl ProbabilityDensityFunctionSampling<1> for &NormalPDF {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        let normal = Normal::new(self.mean, self.std_dev).expect("invalid variance");

        pmatrix.iter_mut().try_for_each(|col| {
            *col = match self.range {
                Some((min, max)) => {
                    let mut candidate = rng.sample(normal);

                    let mut limit_counter = 0;

                    // Continsouly draw candidates until a sample is drawn within the valid range.
                    while ((min > candidate) | (candidate > max)) && limit_counter < 100 {
                        candidate = rng.sample(normal);
                        limit_counter += 1;
                    }

                    if limit_counter == 50 {
                        return Err(StatsError::ReachedSamplerLimit(50));
                    } else {
                        candidate
                    }
                }
                None => rng.sample(normal),
            };

            Ok(())
        })?;

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range.unwrap_or((
            self.mean - 3.0 * self.std_dev,
            self.mean + 3.0 * self.std_dev,
        ))]
    }
}

/// A reciprocal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReciprocalPDF {
    pub range: (f32, f32),
}

impl ProbabilityDensityFunctionSampling<1> for &ReciprocalPDF {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        let (minv, maxv) = self.range;

        // Inverse transform sampling.
        let ratio = maxv / minv;
        let cdf_inv = |u: f32| minv * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(0.0, 1.0).unwrap();

        pmatrix
            .iter_mut()
            .for_each(|col| *col = cdf_inv(rng.sample(uniform)));

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}

/// A uniform probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UniformPDF {
    pub range: (f32, f32),
}

impl ProbabilityDensityFunctionSampling<1> for &UniformPDF {
    fn sample_fill<RStride: Dim, CStride: Dim>(
        &self,
        pmatrix: &mut PMatrixViewMut<U1, Dyn, RStride, CStride>,
        rng: &mut (impl Rng + Clone),
    ) -> Result<(), StatsError> {
        let (minv, maxv) = self.range;
        let uniform = Uniform::new_inclusive(minv, maxv).unwrap();

        pmatrix
            .iter_mut()
            .for_each(|col| *col = rng.sample(uniform));

        Ok(())
    }

    fn valid_range(&self) -> [(Fp, Fp); 1] {
        [self.range]
    }
}
