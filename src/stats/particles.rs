use crate::{
    math::CovMatrix,
    stats::{Density, DensityRange, MultivariateNormalDensity},
};
use nalgebra::{
    Const, DVector, Dyn, Matrix, MatrixView, RealField, SVector, SVectorView, U1, VecStorage,
    ViewStorage,
    iter::{ColumnIter, ColumnIterMut},
    par_iter::{ParColumnIter, ParColumnIterMut},
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Mul, MulAssign},
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

    /// Ensemble weights.
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
        opt_range: Option<&SVector<DensityRange<T>, D>>,
        opt_weights: Option<DVector<T>>,
    ) -> Option<Self>
    where
        T: Sum,
        StandardNormal: Distribution<T>,
        usize: AsPrimitive<T>,
    {
        let mvpdf = match &opt_weights {
            Some(weights) => MultivariateNormalDensity::from_vectors::<U1, Const<D>>(
                &vectors.as_view(),
                match opt_range {
                    Some(range) => *range,
                    None => SVector::from([DensityRange::inf(); D]),
                },
                Some(weights.as_slice()),
            ),
            None => MultivariateNormalDensity::from_vectors::<U1, Const<D>>(
                &vectors.as_view(),
                match opt_range {
                    Some(range) => *range,
                    None => SVector::from([DensityRange::inf(); D]),
                },
                None,
            ),
        }?;

        Some(Self {
            kde: None,
            mvpdf,
            particles: vectors.clone_owned(),
            range: match opt_range {
                Some(range) => *range,
                None => SVector::from([DensityRange::inf(); D]),
            },
            weights: if let Some(weights) = opt_weights {
                weights.clone()
            } else {
                DVector::from_element(vectors.ncols(), T::one() / vectors.ncols().as_())
            },
        })
    }

    /// Return a view to all values of a model parameter.
    pub fn get_param_values(
        &self,
        index: usize,
    ) -> Matrix<T, U1, Dyn, ViewStorage<T, U1, Dyn, U1, Const<D>>> {
        self.particles.row(index)
    }

    /// Return a view to and individual particle
    pub fn get_particle(&self, index: usize) -> SVectorView<T, D> {
        self.particles.column(index)
    }

    /// Returns true if the density contains no particles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate through all particles of the ensemble .
    pub fn iter(&self) -> ColumnIter<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        self.particles.column_iter()
    }

    /// Mutably iterate through all particles of the ensemble .
    pub fn iter_mut(&mut self) -> ColumnIterMut<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        self.particles.column_iter_mut()
    }

    /// Estimate the Kullback-Leibler divergence between two [`ParticleDensity`] using the multivariate normal estimates.
    pub fn kullback_leibler_divergence(&self, other: &ParticleDensity<T, D>) -> Option<T>
    where
        T: Sum,
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

    /// Iterate through all particles of the ensemble in parallel.
    pub fn par_iter(&self) -> ParColumnIter<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        self.particles.par_column_iter()
    }

    /// Mutably iterate through all particles of the ensemble in parallel.
    pub fn par_iter_mut(
        &mut self,
    ) -> ParColumnIterMut<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        self.particles.par_column_iter_mut()
    }

    /// Draw a single sample (vector) from the ensemble of particles.
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

    /// Return a reference to the underlying particle matrix.
    pub fn particles(&self) -> &Matrix<T, Const<D>, Dyn, VecStorage<T, Const<D>, Dyn>> {
        &self.particles
    }

    /// Set the kernel density estimator covariance matrix.
    pub fn set_kde_covmatrix(&mut self) {
        self.kde = Some(self.mvpdf.generate_kde_covmatrix(self.particles.ncols()));
    }

    /// Set an individual particle.
    pub fn set_particle(&mut self, index: usize, vector: &SVectorView<T, D>) {
        self.particles.set_column(index, vector)
    }

    /// Update estimated [`CovMatrix`] from the ensemble of particles.
    pub fn update_mvpdf(&mut self)
    where
        T: Sum,
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

    /// Update the particle weights.
    pub fn update_weights(&mut self, weights: &[T]) {
        assert!(
            self.len() == weights.len(),
            "number of weights must be equal to the ensemble size"
        );

        self.weights = DVector::from_iterator(self.len(), weights.iter().copied());
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &DVector<T> {
        &self.weights
    }
}

impl<T, const D: usize> Density<T, D> for &ParticleDensity<T, D>
where
    T: Copy + RealField + SampleUniform + Sum,
    StandardNormal: Distribution<T>,
{
    fn draw_sample(&self, rng: &mut impl Rng, max_attempts: usize) -> Option<SVector<T, D>> {
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

        self.mvpdf
            .draw_sample_with_offset(&offset, rng, max_attempts)
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
            SVector::from([DensityRange::new(-0.75, 0.75); 2]),
            None,
        )
        .unwrap();

        let mut ptpdf_0 = ParticleDensity::from_vectors(
            &array_0.as_view(),
            Some(&SVector::from([DensityRange::new(-0.75, 0.75); 2])),
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
            Some(&SVector::from([DensityRange::new(-0.75, 0.75); 3])),
            None,
        )
        .unwrap();

        ptpdf.set_kde_covmatrix();

        assert!(
            !ptpdf
                .density(&SVector::from([0.2, -0.15, 0.35]).as_view())
                .is_finite()
        );

        assert!(ulps_eq!(
            ptpdf.density(&SVector::from([0.2, 0.0, 0.35]).as_view()),
            0.15514776
        ));

        assert!(ulps_eq!(
            (&ptpdf).draw_sample(&mut rng, 100).unwrap(),
            SVector::from([-0.51040643, 0.0, -0.25138772])
        ));
    }
}
