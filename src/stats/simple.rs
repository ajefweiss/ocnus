use crate::stats::{Density, DensityRange};
use derive_more::{Deref, DerefMut, IntoIterator};
use log::error;
use nalgebra::{MatrixView, RealField, SVector, SVectorView, Scalar, U1};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A joint probability density function composed of `N` independent univariate density functions.
///
/// It is generally recommended to use this density as a model prior, as one can easily fine tune parameters for each dimension.
///
/// ```
/// # use ocnus::stats::{ConstantDensity, MultivariateDensity, UniformDensity, ReciprocalDensity};
/// let prior = MultivariateDensity::new([
///     UniformDensity::new((-1.5, 1.5)).unwrap(),
///     ConstantDensity::new(1.0),
///     ReciprocalDensity::new((0.1, 0.35)).unwrap(),
///  ]);
/// ```

#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct MultivariateDensity<T, D>(
    #[into_iterator(owned, ref, ref_mut)]
    #[serde(with = "serde_arrays")]
    #[serde(bound = "T: for<'x> Deserialize<'x> + Serialize")]
    [UnivariateDensity<T>; N],
)
where
    T: Clone + Scalar;

impl<T, D> MultivariateDensity<T, N>
where
    T: Clone + Scalar,
{
    /// Create a new [`MultivariateDensity`].
    pub fn new(uvpdfs: [UnivariateDensity<T>; N]) -> Self {
        Self(uvpdfs)
    }
}

impl<T, D> Density<T, N> for &MultivariateDensity<T, N>
where
    T: Copy + RealField + SampleUniform + Scalar,
    StandardNormal: Distribution<T>,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<OVector<T, D>> {
        let mut sample = [T::zero(); N];

        for i in 0..N {
            sample[i] = match (&self.0[i]).draw_sample(rng) {
                Some(sample) => sample[0],
                None => return None,
            };
        }

        Some(SVector::from(sample))
    }

    fn relative_density(&self, x: &VectorView<T, D>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let mut rlh = T::one();

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([*value]);
            rlh *= uvpdf.relative_density(&vec.as_view());
        });

        rlh
    }

    fn get_constants(&self) -> [Option<T>; N] {
        self.0
            .iter()
            .map(|uvpdf| uvpdf.get_constants()[0])
            .collect::<Vec<Option<T>>>()
            .try_into()
            .unwrap()
    }

    fn get_range(&self) -> SVector<DensityRange<T>, N> {
        SVector::from_iterator(self.0.iter().map(|uvpdf| uvpdf.get_range()[0]))
    }
}

/// An algebraic data type for univariate probability density functions.
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum UnivariateDensity<T>
where
    T: Clone + Scalar,
{
    Constant(ConstantDensity<T>),
    Cosine(CosineDensity<T>),
    Normal(NormalDensity<T>),
    Reciprocal(ReciprocalDensity<T>),
    Uniform(UniformDensity<T>),
}

impl<T> Density<T, 1> for &UnivariateDensity<T>
where
    T: Copy + RealField + SampleUniform + Scalar,
    StandardNormal: Distribution<T>,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        let sample = match self {
            UnivariateDensity::Constant(pdf) => pdf.draw_sample(rng),
            UnivariateDensity::Cosine(pdf) => pdf.draw_sample(rng),
            UnivariateDensity::Normal(pdf) => pdf.draw_sample(rng),
            UnivariateDensity::Reciprocal(pdf) => pdf.draw_sample(rng),
            UnivariateDensity::Uniform(pdf) => pdf.draw_sample(rng),
        }?;

        Some(sample)
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        match self {
            UnivariateDensity::Constant(pdf) => pdf.relative_density(x),
            UnivariateDensity::Cosine(pdf) => pdf.relative_density(x),
            UnivariateDensity::Normal(pdf) => pdf.relative_density(x),
            UnivariateDensity::Reciprocal(pdf) => pdf.relative_density(x),
            UnivariateDensity::Uniform(pdf) => pdf.relative_density(x),
        }
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [None; 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        match self {
            UnivariateDensity::Constant(pdf) => pdf.get_range(),
            UnivariateDensity::Cosine(pdf) => pdf.get_range(),
            UnivariateDensity::Normal(pdf) => pdf.get_range(),
            UnivariateDensity::Reciprocal(pdf) => pdf.get_range(),
            UnivariateDensity::Uniform(pdf) => pdf.get_range(),
        }
    }
}

/// A constant probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConstantDensity<T>
where
    T: Clone + Scalar,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> ConstantDensity<T>
where
    T: Copy + PartialOrd + Scalar,
{
    /// Create a new [`UnivariateDensity`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(constant: T) -> UnivariateDensity<T> {
        UnivariateDensity::Constant(Self {
            range: SVector::from([DensityRange::new((constant, constant))]),
        })
    }
}

impl<T> Density<T, 1> for &ConstantDensity<T>
where
    T: Copy + RealField + Scalar,
{
    fn draw_sample(&self, _rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        Some(SVector::from([self.range[0].min()]))
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [Some(self.range[0].min()); 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }
}

/// A cosine probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CosineDensity<T>
where
    T: Clone + Scalar,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> CosineDensity<T>
where
    T: Copy + RealField + Scalar,
{
    /// Create a new [`UnivariateDensity`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(minamax: (T, T)) -> Option<UnivariateDensity<T>> {
        let range = DensityRange::new(minamax);

        if (range.min() < -T::frac_pi_2())
            || (range.max() > T::frac_pi_2())
            || (range.min() > range.max())
        {
            None
        } else {
            Some(UnivariateDensity::Cosine(Self {
                range: SVector::from([range]),
            }))
        }
    }
}

impl<T> Density<T, 1> for &CosineDensity<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.

        let uniform =
            Uniform::new_inclusive(self.range[0].min().sin(), self.range[0].max().sin()).unwrap();

        Some(SVector::from([rng.sample(uniform).asin()]))
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        x[0].cos()
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [None; 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }
}

/// A univariate normal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NormalDensity<T>
where
    T: Clone + Scalar,
{
    mean: T,
    range: SVector<DensityRange<T>, 1>,
    std_dev: T,
}

impl<T> NormalDensity<T>
where
    T: Copy + PartialOrd + Scalar,
{
    /// Create a new [`UnivariateDensity`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(mean: T, std_dev: T, minamax: (T, T)) -> Option<UnivariateDensity<T>> {
        let range = DensityRange::new(minamax);

        if (range.min() > mean) || (range.max() < mean) || (range.min() > range.max()) {
            None
        } else {
            Some(UnivariateDensity::Normal(Self {
                mean,
                std_dev,
                range: SVector::from([range]),
            }))
        }
    }
}

impl<T> Density<T, 1> for &NormalDensity<T>
where
    T: Copy + RealField + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        let normal = StandardNormal;

        let sample = {
            let mut attempts = 0;
            let mut candidate = self.std_dev * rng.sample(normal) + self.mean;

            // Continsouly draw candidates until a sample is drawn within the valid range.
            while (self.range[0].min() > candidate) | (candidate > self.range[0].max()) {
                candidate = rng.sample(normal);

                attempts += 1;

                if attempts > 99 {
                    error!(
                        "NormalDensity::draw_sample has failed to draw a valid sample after {} tries",
                        attempts
                    );

                    return None;
                }
            }

            candidate
        };

        Some(SVector::from([sample]))
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        ((x[0] - self.mean) / self.std_dev).powi(2) / T::from_usize(2).unwrap()
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [None; 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }
}

/// A reciprocal probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReciprocalDensity<T>
where
    T: Clone + Scalar,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> ReciprocalDensity<T>
where
    T: Copy + RealField + SampleUniform,
{
    /// Create a new [`UnivariateDensity`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(minamax: (T, T)) -> Option<UnivariateDensity<T>> {
        let range = DensityRange::new(minamax);

        if (range.min() <= T::zero()) || (range.max() < T::zero()) || (range.min() > range.max()) {
            None
        } else {
            Some(UnivariateDensity::Reciprocal(Self {
                range: SVector::from([range]),
            }))
        }
    }
}

impl<T> Density<T, 1> for &ReciprocalDensity<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        if (self.range[0].min() < T::zero()) || (self.range[0].max() < T::zero()) {
            return None;
        }

        // Inverse transform sampling.
        let ratio = self.range[0].max() / self.range[0].min();
        let cdf_inv = |u: T| self.range[0].min() * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(T::zero(), T::one()).unwrap();

        Some(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one() / (x[0] * (self.range[0].max().ln() - self.range[0].min().ln()))
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [None; 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }
}

/// A uniform probability density function.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct UniformDensity<T>
where
    T: Clone + Scalar,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> UniformDensity<T>
where
    T: Copy + RealField,
{
    /// Create a new [`UnivariateDensity`].
    #[allow(clippy::new_ret_no_self)]
    pub fn new(minamax: (T, T)) -> Option<UnivariateDensity<T>> {
        let range = SVector::from([DensityRange::new(minamax)]);

        if range[0].min() >= range[0].max() {
            None
        } else {
            Some(UnivariateDensity::Uniform(Self { range }))
        }
    }
}

impl<T> Density<T, 1> for &UniformDensity<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn draw_sample(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        let uniform = Uniform::new_inclusive(self.range[0].min(), self.range[0].max()).unwrap();

        Some(SVector::from([rng.sample(uniform)]))
    }

    fn relative_density(&self, x: &MatrixView<T, U1, U1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }

    fn get_constants(&self) -> [Option<T>; 1] {
        [None; 1]
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_simple() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let uvpdf = &MultivariateDensity::new([
            ConstantDensity::new(1.0),
            CosineDensity::new((0.1, 0.2)).unwrap(),
            NormalDensity::new(0.1, 0.25, (-0.5, 1.5)).unwrap(),
            ReciprocalDensity::new((0.1, 0.5)).unwrap(),
            UniformDensity::new((1.0, 2.0)).unwrap(),
        ]);

        assert!(ulps_eq!(
            uvpdf.relative_density(&SVector::from([1.0, 0.15, 0.15, 0.2, 1.5]).as_view()),
            0.0614358f32
        ));

        assert!(
            (&uvpdf)
                .relative_density(&SVector::from([1.0, 0.05, 0.15, 0.2, 1.5]).as_view())
                .is_nan()
        );

        assert!(ulps_eq!(
            (&uvpdf).draw_sample(&mut rng).unwrap(),
            SVector::from([1.0, 0.1810371, 0.2788901, 0.1174904, 1.7462168,])
        ));

        assert!((&uvpdf).validate_sample(&(&uvpdf).draw_sample(&mut rng).unwrap().as_view()));
    }
}
