use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

use crate::{
    base::{OcnusEnsbl, OcnusModel, OcnusModelError, ScObs, ScObsSeries},
    obser::{NullNoise, ObserVec},
};
use covmatrix::CovMatrix;
use nalgebra::{Const, DMatrix, Dim, RealField, SMatrix, SVector, VectorView};
use num_traits::AsPrimitive;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

/// Generic method that computes the fisher information matrix (FIM) for an observation function that
/// returns vector observables and with a time/distance dependent auto-correlation function.
pub fn fisher_information_matrix<
    M,
    T,
    const D: usize,
    const N: usize,
    OF,
    CF,
    FMST,
    CSST,
    RStride: Dim,
    CStride: Dim,
>(
    model: &M,
    series: &ScObsSeries<T>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    obs_func: &OF,
    acr_func: &CF,
) -> Result<SMatrix<T, D, D>, OcnusModelError<T>>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy
        + RealField
        + SampleUniform
        + for<'x> Mul<&'x T, Output = T>
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    OF: Fn(&ScObs<T>, &SVector<T, D>, &FMST, &CSST) -> Result<ObserVec<T, N>, OcnusModelError<T>>
        + Sync,
    CF: Fn(T, T) -> T + Sync,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    let mut result = SMatrix::<T, D, D>::zeros();
    let step_sizes = model.param_step_sizes();

    let mut pos = OcnusEnsbl::new(D, model.get_range());
    let mut neg = OcnusEnsbl::new(D, model.get_range());

    let mut pos_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), D);
    let mut neg_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), D);

    pos.ptpdf
        .particles_mut()
        .column_iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] += step_sizes[pdx];
        });

    neg.ptpdf
        .particles_mut()
        .column_iter_mut()
        .enumerate()
        .for_each(|(pdx, mut column)| {
            column.set_column(0, params);
            column[pdx] -= step_sizes[pdx];
        });

    model.initialize_states_ensbl(&mut pos)?;
    model.initialize_states_ensbl(&mut neg)?;

    model.simulate_ensbl(
        series,
        &mut pos,
        obs_func,
        &mut pos_output.as_view_mut(),
        None::<&mut NullNoise<T>>,
    )?;
    model.simulate_ensbl(
        series,
        &mut neg,
        obs_func,
        &mut neg_output.as_view_mut(),
        None::<&mut NullNoise<T>>,
    )?;

    result
        .row_iter_mut()
        .enumerate()
        .for_each(|(rdx, mut row)| {
            row.iter_mut().enumerate().for_each(|(cdx, value)| {
                let dmu_a = pos_output
                    .row_iter()
                    .zip(neg_output.row_iter())
                    .map(|(pos_col, neg_col)| {
                        if step_sizes[rdx] == T::zero() {
                            &pos_col[rdx] - &neg_col[rdx]
                        } else {
                            (&pos_col[rdx] - &neg_col[rdx])
                                / (T::from_usize(2).unwrap() * step_sizes[rdx])
                        }
                    })
                    .collect::<Vec<ObserVec<T, N>>>();

                let dmu_b = pos_output
                    .row_iter()
                    .zip(neg_output.row_iter())
                    .map(|(pos_col, neg_col)| {
                        if step_sizes[cdx] == T::zero() {
                            &pos_col[rdx] - &neg_col[rdx]
                        } else {
                            (&pos_col[cdx] - &neg_col[cdx])
                                / (T::from_usize(2).unwrap() * step_sizes[cdx])
                        }
                    })
                    .collect::<Vec<ObserVec<T, N>>>();

                // Normalization procedure.
                // TODO: This is not required once proper cov-matrices exist.
                let obs_norm_a = dmu_a.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);
                let obs_norm_b = dmu_b.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);

                let valid_indices = dmu_a.iter().map(|x| !x.any_nan()).collect::<Vec<bool>>();

                let dmu_a_mat = DMatrix::from_iterator(
                    N,
                    obs_norm_a,
                    dmu_a
                        .iter()
                        .filter_map(|obsvec| {
                            if !obsvec.any_nan() {
                                Some(obsvec.into_iter().copied())
                            } else {
                                None
                            }
                        })
                        .flatten(),
                );

                let dmu_b_mat = DMatrix::from_iterator(
                    N,
                    obs_norm_b,
                    dmu_b
                        .iter()
                        .filter_map(|obsvec| {
                            if !obsvec.any_nan() {
                                Some(obsvec.into_iter().copied())
                            } else {
                                None
                            }
                        })
                        .flatten(),
                );

                let coviter =
                    (0..valid_indices.len())
                        .zip(series)
                        .filter_map(|(i, scobs_i)| {
                            if valid_indices[i] {
                                Some((0..valid_indices.len()).zip(series).filter_map(
                                    |(j, scobs_j)| {
                                        if valid_indices[j] {
                                            Some(acr_func(
                                                (*scobs_i.get_timestamp()
                                                    - *scobs_j.get_timestamp())
                                                .abs(),
                                                scobs_i.distance(scobs_j),
                                            ))
                                        } else {
                                            None
                                        }
                                    },
                                ))
                            } else {
                                None
                            }
                        })
                        .flatten();

                let covariance = CovMatrix::new(
                    DMatrix::<T>::from_iterator(obs_norm_a, obs_norm_b, coviter),
                    true,
                )
                .unwrap();

                *value = (0..N)
                    .map(|idx| {
                        (dmu_a_mat.row(idx)
                            * covariance.pseudo_inverse()
                            * dmu_b_mat.row(idx).transpose())[(0, 0)]
                    })
                    .sum::<T>();
            })
        });

    // Fill out other matrix half.
    result += result.transpose() - SMatrix::<T, D, D>::from_diagonal(&result.diagonal());

    Ok(result)
}
