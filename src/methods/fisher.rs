use crate::{
    base::{Model, ModelEnsbl, ModelError, Obser, ScConf, ScObs},
    math::CovMatrix,
    obsty::{NullNoise, ObserVec},
    stats::Density,
};
use nalgebra::{Const, DMatrix, Dim, Dyn, RealField, SMatrix, SVector, VectorView};
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use std::iter::Sum;

/// Generic method that computes the fisher information matrix (FIM) for an observation function that
/// returns vector observables and with a given covariance matrix.
///
/// The covariance matrix must have an appropriate size w.r.t. to the observation scobs and all observations must be valid.
pub fn fisher_information_matrix<
    M,
    T,
    const D: usize,
    const N: usize,
    OF,
    RStride: Dim,
    CStride: Dim,
>(
    model: &M,
    scobs: &ScObs<T, ObserVec<T, N>>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    obs_func: &OF,
    covm: &CovMatrix<T, Dyn>,
) -> Result<SMatrix<T, D, D>, ModelError<T>>
where
    T: Copy + Float + RealField + SampleUniform + Sum,
    M: Model<T, D> + ?Sized + Sync,
    OF: Fn(
            &M,
            &ScConf<T>,
            &SVector<T, D>,
            &M::FMST,
            &M::CSST,
        ) -> Result<ObserVec<T, N>, ModelError<T>>
        + Sync,
    M::CSST: Clone + Default + Send,
    M::FMST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    let mut result = SMatrix::<T, D, D>::zeros();

    let mut pos = ModelEnsbl::<T, M, D>::new(D, None);
    let mut neg = ModelEnsbl::<T, M, D>::new(D, None);

    let mut obser_pos = Obser::<T, ObserVec<T, N>>::new(scobs.clone(), D);
    let mut obser_neg = Obser::<T, ObserVec<T, N>>::new(scobs.clone(), D);

    let step_sizes = SVector::<T, D>::from_iterator(
        model
            .model_prior()
            .get_range()
            .iter()
            .map(|range| range.length() * T::from_usize(1024).unwrap() * T::epsilon()),
    );

    pos.ptpdf
        .iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] += step_sizes[pdx];
        });

    neg.ptpdf
        .iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] -= step_sizes[pdx];
        });

    model.initialize_states_ensbl(&mut pos)?;
    model.initialize_states_ensbl(&mut neg)?;

    model.simulate_ensbl(
        &mut pos,
        &mut obser_pos,
        obs_func,
        &mut None::<&mut NullNoise<T>>,
    )?;

    model.simulate_ensbl(
        &mut neg,
        &mut obser_neg,
        obs_func,
        &mut None::<&mut NullNoise<T>>,
    )?;

    // // Only allow non-NaN outs.
    // obser_pos.output().iter().try_for_each(|value| {
    //     if value.is_valid() {
    //         Ok(())
    //     } else {
    //         Err(ModelError::OutputNaN)
    //     }
    // })?;

    // obser_neg.output().iter().try_for_each(|value| {
    //     if value.is_valid() {
    //         Ok(())
    //     } else {
    //         Err(ModelError::OutputNaN)
    //     }
    // })?;

    result
        .row_iter_mut()
        .enumerate()
        .for_each(|(rdx, mut row)| {
            row.iter_mut().enumerate().for_each(|(cdx, value)| {
                if cdx <= rdx {
                    let dmu_a = obser_pos
                        .time_iter_mut()
                        .zip(obser_neg.time_iter_mut())
                        .map(|((_, _, pos_col), (_, _, neg_col))| {
                            if step_sizes[rdx] == T::zero() {
                                ObserVec::zeros()
                            } else {
                                (&pos_col[rdx] - &neg_col[rdx])
                                    / (T::from_usize(2).unwrap() * step_sizes[rdx])
                            }
                        })
                        .collect::<Vec<ObserVec<T, N>>>();

                    let dmu_b = obser_pos
                        .time_iter_mut()
                        .zip(obser_neg.time_iter_mut())
                        .map(|((_, _, pos_col), (_, _, neg_col))| {
                            if step_sizes[cdx] == T::zero() {
                                ObserVec::zeros()
                            } else {
                                (&pos_col[cdx] - &neg_col[cdx])
                                    / (T::from_usize(2).unwrap() * step_sizes[cdx])
                            }
                        })
                        .collect::<Vec<ObserVec<T, N>>>();

                    let dmu_a_mat = DMatrix::from_iterator(
                        N,
                        scobs.len(),
                        dmu_a.iter().flat_map(|obsvec| obsvec.into_iter().copied()),
                    );

                    let dmu_b_mat = DMatrix::from_iterator(
                        N,
                        scobs.len(),
                        dmu_b.iter().flat_map(|obsvec| obsvec.into_iter().copied()),
                    );

                    *value = (0..N)
                        .map(|idx| {
                            (dmu_a_mat.row(idx)
                                * covm.pseudo_inverse()
                                * dmu_b_mat.row(idx).transpose())[(0, 0)]
                        })
                        .sum::<T>();
                }
            })
        });

    // Fill other matrix half.
    result += result.transpose() - SMatrix::<T, D, D>::from_diagonal(&result.diagonal());

    Ok(result)
}
