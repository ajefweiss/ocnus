use crate::stats::{Density, DensityRange, MultivariateNormalDensity};
use covmatrix::CovMatrix;
use nalgebra::{
    Const, DVector, Dyn, Matrix, MatrixView, RealField, SVector, SVectorView, U1, VecStorage,
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Mul, MulAssign, Sub},
};

/// A probability density function defined by an ensemble of particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct ParticleDensity<T, const D: usize>
where
    T: Copy + RealField,
{
    /// The covariance matrix that is used for kernel density estimation.
    kde: Option<CovMatrix<T, Const<D>>>,

    /// An [`MultivariateNormalDensity`] estimate of the underlying density.
    mvpdf: MultivariateNormalDensity<T, D>,

    /// The particle ensemble that approximates the underlying density.
    particles: Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>>,

    /// Valid parameter range.
    range: SVector<DensityRange<T>, D>,

    /// Particle ensemble weights
    weights: DVector<T>,
}

impl<T, const D: usize> ParticleDensity<T, D>
where
    T: Copy + RealField,
{
    /// Returns a reference to the underlying estimated [`CovMatrix`].
    pub fn covmatrix(&self) -> &CovMatrix<T, Const<D>> {
        self.mvpdf.covmatrix()
    }

    /// Estimates the exact density  at a specific position `x`.
    pub fn density(&self, x: &SVectorView<T, D>) -> T
    where
        T: SampleUniform + Sum,
        StandardNormal: Distribution<T>,
    {
        let density_rel = self.relative_density(x);

        let normalization = T::one()
            / T::two_pi().powf(
                T::from_usize(self.kde.as_ref().unwrap().rank()).unwrap()
                    / T::from_usize(2).unwrap(),
            )
            / self
                .kde
                .as_ref()
                .unwrap()
                .pseudo_determinant()
                .unwrap()
                .sqrt();

        normalization * density_rel
    }

    /// Create a [`ParticleDensity`] from an ensemble of particles.
    pub fn from_vectors(
        vectors: &MatrixView<T, Const<D>, Dyn>,
        range: SVector<DensityRange<T>, D>,
        opt_weights: Option<DVector<T>>,
    ) -> Option<Self>
    where
        T: for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let mvpdf = match &opt_weights {
            Some(weights) => MultivariateNormalDensity::from_vectors::<U1, Const<D>>(
                &vectors.as_view(),
                range,
                Some(weights.as_slice()),
            ),
            None => MultivariateNormalDensity::from_vectors::<U1, Const<D>>(
                &vectors.as_view(),
                range,
                None,
            ),
        }?;

        Some(Self {
            kde: None,
            mvpdf,
            particles: vectors.clone_owned(),
            range,
            weights: if let Some(weights) = opt_weights {
                weights.clone()
            } else {
                DVector::from_element(vectors.ncols(), T::one() / vectors.ncols().as_())
            },
        })
    }

    /// Update `self` assuming a transition
    /// from another [`ParticleDensity`] with optional prior.
    pub fn from_transition<P>(&mut self, other: &Self, opt_prior: Option<&P>)
    where
        T: for<'x> Mul<&'x T, Output = T>
            + SampleUniform
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        P: Density<T, D>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let covmat_inv = other.mvpdf.pseudo_inverse();

        let mut weights_vector = vec![T::zero(); self.particles.ncols()];

        weights_vector
            .par_iter_mut()
            .zip(self.particles.par_column_iter())
            .for_each(|(weight, params_new)| {
                *weight = {
                    let value = other
                        .particles
                        .column_iter()
                        .zip(other.weights.iter())
                        .map(|(params_old, weight_old)| {
                            let delta = params_new - params_old;

                            (weight_old.ln() - (delta.transpose() * covmat_inv * delta)[(0, 0)])
                                .exp()
                        })
                        .sum::<T>();

                    if let Some(prior) = opt_prior {
                        (T::one() / value) * prior.relative_density(&params_new)
                    } else {
                        T::one() / value
                    }
                }
            });

        let mut weights = DVector::from(weights_vector);

        let weights_total = weights.iter().sum::<T>();

        weights
            .iter_mut()
            .for_each(|weight| *weight /= weights_total);

        let mvpdf = MultivariateNormalDensity::from_vectors::<_, _>(
            &self.particles.as_view::<_, _, U1, Dyn>(),
            self.range,
            Some(weights.as_slice()),
        )
        .unwrap();

        self.mvpdf = mvpdf;
        self.weights = weights;

        if self.kde.is_some() {
            self.set_kde_covmatrix();
        }
    }

    /// Returns true if the density contains no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Estimate the Kullback-Leibler divergence between two [`ParticleDensity`] using the multivariate normal estimates.
    pub fn kullback_leibler_divergence(&self, other: &ParticleDensity<T, D>) -> Option<T>
    where
        T: Sum + for<'x> Sum<&'x T>,
    {
        self.mvpdf.kullback_leibler_divergence(&other.mvpdf)
    }

    /// Returns the number of particles, also referred to as its 'length'.
    pub fn len(&self) -> usize {
        self.particles.ncols()
    }

    /// Returns the estimated mean of the density.
    pub fn mean(&self) -> &SVector<T, D> {
        self.mvpdf.mean()
    }

    /// Returns a reference to the particles matrix.
    pub fn particles(&self) -> &Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        &self.particles
    }

    /// Returns a reference to the particles matrix.
    pub fn particles_mut(&mut self) -> &mut Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        &mut self.particles
    }

    /// Resample from existing particles.
    pub fn resample(&self, rng: &mut impl Rng) -> SVector<T, D>
    where
        T: SampleUniform,
    {
        let uniform = Uniform::new(T::zero(), T::one()).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx: T = rng.sample(uniform);

                // Here we abuse try_fold to return particle index early wrapped within Err().
                match self
                    .weights
                    .iter()
                    .enumerate()
                    .try_fold(T::zero(), |acc, (idx, weight)| {
                        let next_weight = acc + *weight;
                        if wdx < next_weight {
                            Err(idx)
                        } else {
                            Ok(next_weight)
                        }
                    }) {
                    Ok(_) => self.weights.len() - 1,
                    Err(idx) => idx,
                }
            };

            self.particles.column(pdx)
        };

        offset.into_owned()
    }

    /// Set the kernel density estimator covariance matrix.
    pub fn set_kde_covmatrix(&mut self) {
        self.kde = Some(self.mvpdf.generate_kde_covmatrix(self.particles.ncols()));
    }

    /// Update estimated [`CovMatrix`].
    pub fn update_mvpdf(&mut self)
    where
        T: for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let mvpdf = MultivariateNormalDensity::from_vectors::<U1, Const<D>>(
            &self.particles.as_view(),
            self.range,
            Some(self.weights.as_slice()),
        )
        .expect("failed to update mvpdf");

        self.mvpdf = mvpdf;
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &DVector<T> {
        &self.weights
    }

    /// Returns a mutable reference to the particle weights.
    pub fn weights_mut(&mut self) -> &mut DVector<T> {
        &mut self.weights
    }
}

impl<T, const D: usize> Density<T, D> for &ParticleDensity<T, D>
where
    T: Copy + RealField + SampleUniform + Sum,
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<SVector<T, D>>
    where
        StandardNormal: Distribution<T>,
    {
        let uniform = Uniform::new(T::zero(), T::one()).unwrap();

        let offset = {
            let pdx = {
                // Select particle index by weight.
                let wdx = rng.sample(uniform);

                // Here we abuse try_fold to return particle index early wrapped within Err().
                match self
                    .weights
                    .iter()
                    .enumerate()
                    .try_fold(T::zero(), |acc, (idx, weight)| {
                        let next_weight = acc + *weight;
                        if wdx < next_weight {
                            Err(idx)
                        } else {
                            Ok(next_weight)
                        }
                    }) {
                    Ok(_) => self.weights.len() - 1,
                    Err(idx) => idx,
                }
            };

            self.particles.column(pdx)
        };

        self.mvpdf.draw_sample_with_offset::<A>(&offset, rng)
    }

    fn get_constants(&self) -> SVector<T, D> {
        (&self.mvpdf).get_constants()
    }

    fn get_range(&self) -> SVector<DensityRange<T>, D> {
        self.range
    }

    fn relative_density(&self, x: &SVectorView<T, D>) -> T {
        if !self.validate_sample(&x.as_view()) {
            return (-T::one()).sqrt();
        }

        self.particles
            .column_iter()
            .zip(self.weights.iter())
            .map(|(col, weight)| {
                *weight
                    * (-self
                        .kde
                        .as_ref()
                        .expect("kde covmatrix is not set")
                        .mahalanobis_distance::<U1, Dyn>(&(x - col).as_view())
                        / T::from_usize(2).unwrap())
                    .exp()
            })
            .sum::<T>()
    }
}

impl<T, const D: usize> Mul<T> for ParticleDensity<T, D>
where
    T: Copy + RealField,
{
    type Output = ParticleDensity<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            kde: self.kde,
            mvpdf: self.mvpdf * rhs,
            particles: self.particles,
            range: self.range,
            weights: self.weights,
        }
    }
}

impl<T, const D: usize> MulAssign<T> for ParticleDensity<T, D>
where
    T: Copy + RealField,
{
    fn mul_assign(&mut self, rhs: T) {
        self.mvpdf *= rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{Matrix, SVector, U2, U3, VecStorage};
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_particle_density() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array_0 = Matrix::<f32, U2, Dyn, VecStorage<f32, U2, Dyn>>::from_iterator(
            10000,
            (0..20000).map(|idx| {
                if idx % 2 == 0 {
                    0.1 + rng.sample::<f32, StandardNormal>(uniform)
                } else {
                    0.25 + rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let mvpdf_0 = MultivariateNormalDensity::from_vectors::<Dyn, U2>(
            &array_0.as_view(),
            SVector::from([DensityRange((-0.75, 0.75)); 2]),
            None,
        )
        .unwrap();

        let mut ptpdf_0 = ParticleDensity::from_vectors(
            &array_0.as_view(),
            SVector::from([DensityRange((-0.75, 0.75)); 2]),
            None,
        )
        .unwrap();

        ptpdf_0.set_kde_covmatrix();

        assert!(ulps_eq!(
            mvpdf_0.density(&SVector::from([0.2, 0.35]).as_view()),
            0.16128483
        ));

        assert!(ulps_eq!(
            ptpdf_0.density(&SVector::from([0.2, 0.35]).as_view()),
            0.15514776
        ));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let array = Matrix::<f32, U3, Dyn, VecStorage<f32, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 0 {
                    0.1 + rng.sample::<f32, StandardNormal>(uniform)
                } else if idx % 3 == 1 {
                    0.0
                } else {
                    0.25 + rng.sample::<f32, StandardNormal>(uniform)
                }
            }),
        );

        let mut ptpdf = ParticleDensity::from_vectors(
            &array.as_view(),
            SVector::from([DensityRange((-0.75, 0.75)); 3]),
            None,
        )
        .unwrap();

        ptpdf.set_kde_covmatrix();

        assert!(
            ptpdf
                .density(&SVector::from([0.2, -0.15, 0.35]).as_view())
                .is_nan()
        );

        assert!(ulps_eq!(
            ptpdf.density(&SVector::from([0.2, 0.0, 0.35]).as_view()),
            0.15514776
        ));

        assert!(ulps_eq!(
            (&ptpdf).draw_sample::<100>(&mut rng).unwrap(),
            SVector::from([-0.51040643, 0.0, -0.25138772])
        ));
    }
}
