use crate::stats::{Density, DensityRange, MultivariateNormalDensity};
use covmatrix::CovMatrix;
use nalgebra::{
    DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimName, DimSub, Dyn, OMatrix, OVector,
    RealField, Scalar, U1, VectorView, allocator::Allocator,
};
use num_traits::AsPrimitive;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

/// A probability density function defined by an ensemble of particles.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(serialize = "
    T: Serialize, 
    OVector<T, D>: Serialize, 
    OVector<DensityRange<T>, D>: Serialize, 
    OMatrix<T, D, D>: Serialize, 
    OMatrix<T, D, Dyn>: Serialize"))]
#[serde(bound(deserialize = "
    T: Deserialize<'de>, 
    OVector<T, D>: Deserialize<'de>, 
    OVector<DensityRange<T>, D>: Deserialize<'de>, 
    OMatrix<T, D, D>: Deserialize<'de> , 
    OMatrix<T, D, Dyn>: Deserialize<'de>"))]
pub struct ParticleDensity<T, D>
where
    T: Copy + RealField + Scalar,
    D: DimName + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    /// The covariance matrix that is used for kernel density estimation.
    kde: Option<CovMatrix<T, D>>,

    /// An [`MultivariateNormalDensity`] estimate of the underlying density.
    mvpdf: MultivariateNormalDensity<T, D>,

    /// The particle ensemble that approximates the underlying density.
    particles: OMatrix<T, D, Dyn>,

    /// Valid parameter range.
    range: OVector<DensityRange<T>, D>,

    /// Particle ensemble weights
    weights: OVector<T, Dyn>,
}

impl<T, D> ParticleDensity<T, D>
where
    T: Copy + RealField + Scalar,
    D: DimName + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    /// Estimates the exact density  at a specific position `x`.
    pub fn density(&self, x: &VectorView<T, D>) -> T
    where
        T: SampleUniform + Sum,
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
        vectors: OMatrix<T, D, Dyn>,
        range: OVector<DensityRange<T>, D>,
        opt_weights: Option<OVector<T, Dyn>>,
    ) -> Option<Self>
    where
        T: for<'x> Mul<&'x T, Output = T>
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
    {
        let mvpdf = match &opt_weights {
            Some(weights) => MultivariateNormalDensity::from_vectors::<_, _>(
                &vectors.as_view(),
                range.clone(),
                Some(weights.as_slice()),
            ),
            None => {
                MultivariateNormalDensity::from_vectors(&vectors.as_view(), range.clone(), None)
            }
        }?;

        let size = vectors.ncols();

        Some(Self {
            kde: None,
            mvpdf,
            particles: vectors,
            range,
            weights: if let Some(weights) = opt_weights {
                weights.clone()
            } else {
                OVector::from_element_generic(Dyn(size), U1, T::one() / size.as_())
            },
        })
    }

    /// Create a [`ParticleDensity`] from an ensemble of particles assuming a transition
    /// from another [`ParticleDensity`] with optional prior.
    pub fn from_vectors_with_transition<F>(
        vectors: OMatrix<T, D, Dyn>,
        other: &Self,
        opt_prior: Option<&F>,
    ) -> Option<Self>
    where
        T: for<'x> Mul<&'x T, Output = T>
            + SampleUniform
            + for<'x> Sub<&'x T, Output = T>
            + Sum
            + for<'x> Sum<&'x T>,
        for<'x> &'x T: Mul<&'x T, Output = T>,
        usize: AsPrimitive<T>,
        F: Density<T, D>,
        <nalgebra::DefaultAllocator as nalgebra::allocator::Allocator<D, nalgebra::Dyn>>::Buffer<T>:
            std::marker::Sync,
    {
        let covmat_inv = other.mvpdf.pseudo_inverse();

        let mut weights = OVector::<T, Dyn>::from_iterator(
            vectors.ncols(),
            vectors.column_iter().map(|params_new| {
                let value = other
                    .particles
                    .column_iter()
                    .zip(other.weights.iter())
                    .map(|(params_old, weight_old)| {
                        let delta = params_new - params_old;

                        (weight_old.ln() - (delta.transpose() * covmat_inv * delta)[(0, 0)]).exp()
                    })
                    .sum::<T>();

                if let Some(prior) = opt_prior {
                    (T::one() / value) * prior.relative_density(&params_new)
                } else {
                    T::one() / value
                }
            }),
        );

        let weights_total = weights.iter().sum::<T>();

        weights
            .iter_mut()
            .for_each(|weight| *weight /= weights_total);

        Self::from_vectors(vectors, other.get_range(), Some(weights))
    }

    /// Set the kernel density estimator covariance matrix.
    pub fn set_kde_covmatrix(&mut self) {
        self.kde = Some(self.mvpdf.generate_kde_covmatrix(self.particles.ncols()));
    }

    /// Returns a reference to the particle weights.
    pub fn weights(&self) -> &OVector<T, Dyn> {
        &self.weights
    }
}

impl<T, D> Density<T, D> for &ParticleDensity<T, D>
where
    T: Copy + RealField + SampleUniform + Scalar + Sum,
    D: DimName + DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<D>
        + Allocator<U1, D>
        + Allocator<D, D>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<DimMinimum<D, D>, D>
        + Allocator<D, DimMinimum<D, D>>
        + Allocator<DimMinimum<D, D>>
        + Allocator<DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    fn draw_sample<const A: usize>(&self, rng: &mut impl Rng) -> Option<OVector<T, D>> {
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

        self.mvpdf.draw_sample_with_offset::<A, _, _>(&offset, rng)
    }

    fn get_constants(&self) -> OVector<T, D> {
        (&self.mvpdf).get_constants()
    }

    fn get_range(&self) -> OVector<DensityRange<T>, D> {
        self.range.clone()
    }

    fn relative_density<RStride, CStride>(&self, x: &VectorView<T, D, RStride, CStride>) -> T
    where
        RStride: Dim,
        CStride: Dim,
    {
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
                        .mahalanobis_distance(&(x - col).as_view())
                        / T::from_usize(2).unwrap())
                    .exp()
            })
            .sum::<T>()
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
            OVector::from([DensityRange((-0.75, 0.75)); 2]),
            None,
        )
        .unwrap();

        let mut ptpdf_0 = ParticleDensity::from_vectors(
            array_0,
            OVector::from([DensityRange((-0.75, 0.75)); 2]),
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
            array,
            OVector::from([DensityRange((-0.75, 0.75)); 3]),
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
