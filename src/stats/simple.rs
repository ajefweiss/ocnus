use crate::stats::{Density, DensityRange};
use derive_more::{Deref, DerefMut, IntoIterator};
use log::error;
use nalgebra::{RealField, SVector, SVectorView};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A joint probability density function composed of `D` univariate density functions.
///
/// It is generally recommended to use this density as a model prior, as one can easily fine tune parameters for each dimension.
///
/// ```
/// # use nalgebra::Const;
/// # use ocnus::stats::{ConstantDensity, CosineDensity, NormalDensity, MultivariateDensity, ReciprocalDensity, UniformDensity};
/// let prior = MultivariateDensity::<f64, 5>::new(&[
///     ConstantDensity::new(1.0),
///     CosineDensity::new((0.1, 0.2)).unwrap(),
///     NormalDensity::new(0.1, 0.25, (-0.5, 1.5)).unwrap(),
///     ReciprocalDensity::new((0.1, 0.5)).unwrap(),
///     UniformDensity::new((1.0, 2.0)).unwrap(),
/// ]);
/// ```
#[derive(Clone, Debug, Deref, DerefMut, Deserialize, IntoIterator, Serialize)]
pub struct MultivariateDensity<T, const D: usize>(
    #[into_iterator(owned, ref, ref_mut)] SVector<UnivariateDensity<T>, D>,
)
where
    T: Copy + RealField;

impl<T, const D: usize> MultivariateDensity<T, D>
where
    T: Copy + RealField,
{
    /// Create a new [`MultivariateDensity`].
    pub fn new<'a, I>(uvpdfs: I) -> Self
    where
        I: IntoIterator<Item = &'a UnivariateDensity<T>>,
    {
        let vector = SVector::<UnivariateDensity<T>, D>::from_iterator(uvpdfs.into_iter().cloned());

        Self(vector)
    }
}

impl<T, const D: usize> Density<T, D> for &MultivariateDensity<T, D>
where
    T: Copy + RealField + SampleUniform + RealField,
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        let mut sample = SVector::<T, D>::zeros();

        for i in 0..D {
            sample[i] = match Density::<T, 1>::draw_sample::<A>(&(&self.0[i]), rng) {
                Some(sample) => sample[0],
                None => return None,
            };
        }

        Some(sample)
    }

    fn get_constants(&self) -> SVector<T, D> {
        SVector::from_iterator(
            self.0
                .iter()
                .map(|uvpdf| Density::<T, 1>::get_constants(&uvpdf)[0]),
        )
    }

    fn get_range(&self) -> SVector<DensityRange<T>, D> {
        SVector::from_iterator(
            self.0
                .iter()
                .map(|uvpdf| Density::<T, 1>::get_range(&uvpdf)[0]),
        )
    }

    fn relative_density(&self, x: &SVectorView<T, D>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        let mut rlh = T::one();

        self.0.iter().zip(x.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::<T, 1>::from([*value]);

            rlh *= Density::<T, 1>::relative_density(&uvpdf, &vec.as_view());
        });

        rlh
    }
}

/// An algebraic data type for univariate probability density functions.
#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum UnivariateDensity<T>
where
    T: Copy + RealField,
{
    Constant(ConstantDensity<T>),
    Cosine(CosineDensity<T>),
    Normal(NormalDensity<T>),
    Reciprocal(ReciprocalDensity<T>),
    Uniform(UniformDensity<T>),
}

impl<T> Density<T, 1> for &UnivariateDensity<T>
where
    T: Copy + RealField + SampleUniform,
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>>
    where
        StandardNormal: Distribution<T>,
    {
        let sample = match self {
            UnivariateDensity::Constant(pdf) => pdf.draw_sample::<A>(rng),
            UnivariateDensity::Cosine(pdf) => pdf.draw_sample::<A>(rng),
            UnivariateDensity::Normal(pdf) => pdf.draw_sample::<A>(rng),
            UnivariateDensity::Reciprocal(pdf) => pdf.draw_sample::<A>(rng),
            UnivariateDensity::Uniform(pdf) => pdf.draw_sample::<A>(rng),
        }?;

        Some(sample)
    }

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([(-T::one()).sqrt()])
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

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        match self {
            UnivariateDensity::Constant(pdf) => pdf.relative_density(x),
            UnivariateDensity::Cosine(pdf) => pdf.relative_density(x),
            UnivariateDensity::Normal(pdf) => pdf.relative_density(x),
            UnivariateDensity::Reciprocal(pdf) => pdf.relative_density(x),
            UnivariateDensity::Uniform(pdf) => pdf.relative_density(x),
        }
    }
}

/// A constant probability density function.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ConstantDensity<T>
where
    T: Copy + RealField,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> ConstantDensity<T>
where
    T: Copy + PartialOrd + RealField,
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
    T: Copy + RealField + RealField,
{
    fn draw_sample<const A: usize>(&self, _rng: &mut impl Rng) -> Option<SVector<T, 1>> {
        Some(SVector::from([self.range[0].min()]))
    }

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([self.range[0].min()])
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }
}

/// A cosine probability density function.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CosineDensity<T>
where
    T: Copy + RealField,
{
    range: SVector<DensityRange<T>, 1>,
}

impl<T> CosineDensity<T>
where
    T: Copy + RealField + RealField,
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
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>>
    where
        StandardNormal: Distribution<T>,
    {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.

        let uniform =
            Uniform::new_inclusive(self.range[0].min().sin(), self.range[0].max().sin()).unwrap();

        Some(SVector::from([rng.sample(uniform).asin()]))
    }

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([(-T::one()).sqrt()])
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        x[0].cos()
    }
}

/// A univariate normal probability density function.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NormalDensity<T>
where
    T: Copy + RealField,
{
    mean: T,
    range: SVector<DensityRange<T>, 1>,
    std_dev: T,
}

impl<T> NormalDensity<T>
where
    T: Copy + PartialOrd + RealField,
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
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>>
    where
        StandardNormal: Distribution<T>,
    {
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

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([(-T::one()).sqrt()])
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        ((x[0] - self.mean) / self.std_dev).powi(2) / T::from_usize(2).unwrap()
    }
}

/// A reciprocal probability density function.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ReciprocalDensity<T>
where
    T: Copy + RealField,
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
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>>
    where
        StandardNormal: Distribution<T>,
    {
        if (self.range[0].min() < T::zero()) || (self.range[0].max() < T::zero()) {
            return None;
        }

        // Inverse transform sampling.
        let ratio = self.range[0].max() / self.range[0].min();
        let cdf_inv = |u: T| self.range[0].min() * (ratio.ln() * u).exp();
        let uniform = Uniform::new_inclusive(T::zero(), T::one()).unwrap();

        Some(SVector::from([cdf_inv(rng.sample(uniform))]))
    }

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([(-T::one()).sqrt()])
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one() / (x[0] * (self.range[0].max().ln() - self.range[0].min().ln()))
    }
}

/// A uniform probability density function.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UniformDensity<T>
where
    T: Copy + RealField,
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
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, 1>>
    where
        StandardNormal: Distribution<T>,
    {
        let uniform = Uniform::new_inclusive(self.range[0].min(), self.range[0].max()).unwrap();

        Some(SVector::from([rng.sample(uniform)]))
    }

    fn get_constants(&self) -> SVector<T, 1> {
        SVector::from([(-T::one()).sqrt()])
    }

    fn get_range(&self) -> SVector<DensityRange<T>, 1> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, 1>) -> T {
        if !self.validate_sample(x) {
            return (-T::one()).sqrt();
        }

        T::one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_simple_density() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let uvpdf = &MultivariateDensity::<f32, 5>::new(&[
            ConstantDensity::new(1.0),
            CosineDensity::new((0.1, 0.2)).unwrap(),
            NormalDensity::new(0.1, 0.25, (-0.5, 1.5)).unwrap(),
            ReciprocalDensity::new((0.1, 0.5)).unwrap(),
            UniformDensity::new((1.0, 2.0)).unwrap(),
        ]);

        assert!(ulps_eq!(
            uvpdf.relative_density(&SVector::from([1.0f32, 0.15, 0.15, 0.2, 1.5]).as_view()),
            0.0614358f32
        ));

        assert!(
            (&uvpdf)
                .relative_density(&SVector::from([1.0, 0.05, 0.15, 0.2, 1.5]).as_view())
                .is_nan()
        );

        assert!(ulps_eq!(
            (&uvpdf).draw_sample::<100>(&mut rng).unwrap(),
            SVector::from([1.0, 0.1810371, 0.2788901, 0.1174904, 1.7462168,])
        ));

        assert!(
            (&uvpdf).validate_sample(&(&uvpdf).draw_sample::<100>(&mut rng).unwrap().as_view())
        );
    }
}
