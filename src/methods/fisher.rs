use std::{
    iter::Sum,
    ops::{Mul, Sub},
};

use crate::{
    base::{OcnusEnsbl, OcnusModel, OcnusModelError, ScObs, ScObsSeries},
    obser::{NullNoise, ObserVec},
    stats::Density,
};
use covmatrix::CovMatrix;
use nalgebra::{Const, DMatrix, Dim, Dyn, RealField, SMatrix, SVector, VectorView};
use num_traits::{AsPrimitive, Float};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

/// Generic method that computes the fisher information matrix (FIM) for an observation function that
/// returns vector observables and with a time/distance dependent auto-correlation function.
pub fn fisher_information_matrix<
    M,
    T,
    const D: usize,
    const N: usize,
    OF,
    FMST,
    CSST,
    RStride: Dim,
    CStride: Dim,
>(
    model: &M,
    series: &ScObsSeries<T>,
    params: &VectorView<T, Const<D>, RStride, CStride>,
    obs_func: &OF,
    covm: &CovMatrix<T, Dyn>,
) -> Result<SMatrix<T, D, D>, OcnusModelError<T>>
where
    M: OcnusModel<T, D, FMST, CSST>,
    T: Copy
        + Float
        + RealField
        + SampleUniform
        + for<'x> Mul<&'x T, Output = T>
        + for<'x> Sub<&'x T, Output = T>
        + Sum
        + for<'x> Sum<&'x T>,
    for<'x> &'x T: Mul<&'x T, Output = T>,
    OF: Fn(
            &M,
            &ScObs<T>,
            &SVector<T, D>,
            &FMST,
            &CSST,
        ) -> Result<ObserVec<T, N>, OcnusModelError<T>>
        + Sync,
    FMST: Clone + Default + Send,
    CSST: Clone + Default + Send,
    StandardNormal: Distribution<T>,
    usize: AsPrimitive<T>,
{
    let mut result = SMatrix::<T, D, D>::zeros();

    let mut pos = OcnusEnsbl::new(D, model.get_range());
    let mut neg = OcnusEnsbl::new(D, model.get_range());

    let mut pos_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), D);
    let mut neg_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), D);

    let step_sizes =
        SVector::<T, D>::from_iterator(model.model_prior().get_range().iter().map(|range| {
            (range.max() - range.min()) * T::from_usize(1024).unwrap() * T::epsilon()
        }));

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

    // Only allow non-NaN outputs.
    pos_output.iter().try_for_each(|value| {
        if !value.any_nan() {
            Ok(())
        } else {
            Err(OcnusModelError::OutputNaN)
        }
    })?;

    neg_output.iter().try_for_each(|value| {
        if !value.any_nan() {
            Ok(())
        } else {
            Err(OcnusModelError::OutputNaN)
        }
    })?;

    result
        .row_iter_mut()
        .enumerate()
        .for_each(|(rdx, mut row)| {
            row.iter_mut().enumerate().for_each(|(cdx, value)| {
                if cdx <= rdx {
                    let dmu_a = pos_output
                        .row_iter()
                        .zip(neg_output.row_iter())
                        .map(|(pos_col, neg_col)| {
                            if step_sizes[rdx] == T::zero() {
                                ObserVec::zeros()
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
                                ObserVec::zeros()
                            } else {
                                (&pos_col[cdx] - &neg_col[cdx])
                                    / (T::from_usize(2).unwrap() * step_sizes[cdx])
                            }
                        })
                        .collect::<Vec<ObserVec<T, N>>>();

                    let dmu_a_mat = DMatrix::from_iterator(
                        N,
                        series.len(),
                        dmu_a.iter().flat_map(|obsvec| obsvec.into_iter().copied()),
                    );

                    let dmu_b_mat = DMatrix::from_iterator(
                        N,
                        series.len(),
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

    // Fill out other matrix half.
    result += result.transpose() - SMatrix::<T, D, D>::from_diagonal(&result.diagonal());

    Ok(result)
}
