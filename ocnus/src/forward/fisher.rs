use crate::{
    OcnusError, fXX,
    forward::{FSMEnsbl, OcnusFSM},
    math::{CovMatrix, T, abs},
    obser::{ObserVec, ObserVecNoise, ScObsSeries},
};
use nalgebra::{Const, DMatrix, Dyn, Matrix, SMatrix, VecStorage};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

/// A trait that enables the calculation of the Fisher Information Matrix (FIM) for a [`OcnusFSM`]
/// with a [`ObserVec`] as observable type.
pub trait FisherInformation<T, const P: usize, const N: usize, FS, CSST>:
    OcnusFSM<T, ObserVec<T, N>, P, FS, CSST>
where
    T: fXX,
    FS: Clone + Default + Send,
    CSST: Clone + Default + Send,
    Self: Sync,
    StandardNormal: Distribution<T>,
{
    /// Compute the Fisher information matrix (FIM) for an array of model parameters using a
    /// time and distance based auto-correlation function `acfunc`.
    fn fischer_information_matrix<C>(
        &self,
        series: &ScObsSeries<T, ObserVec<T, N>>,
        ensbl: &FSMEnsbl<T, P, FS, CSST>,
        acfunc: &C,
    ) -> Result<Vec<SMatrix<T, P, P>>, OcnusError<T>>
    where
        C: Fn(T, T) -> T + Sync,
    {
        let step_sizes = self.param_step_sizes();

        let mut results = vec![SMatrix::<T, P, P>::zeros(); ensbl.len()];

        ensbl
            .params_array
            .par_column_iter()
            .zip(results.par_iter_mut())
            .chunks(Self::RCS / 4)
            .try_for_each(|mut chunks| {
                chunks.iter_mut().try_for_each(|(params_ref, fim)| {
                    let mut pos = FSMEnsbl {
                        params_array:
                            Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::from_columns(
                                &[*params_ref; P],
                            ),
                        fm_states: vec![FS::default(); P],
                        cs_states: vec![CSST::default(); P],
                        weights: vec![T::one() / T::from_usize(P).unwrap(); P],
                    };

                    let mut neg = FSMEnsbl {
                        params_array:
                            Matrix::<T, Const<P>, Dyn, VecStorage<T, Const<P>, Dyn>>::from_columns(
                                &[*params_ref; P],
                            ),
                        fm_states: vec![FS::default(); P],
                        cs_states: vec![CSST::default(); P],
                        weights: vec![T::one() / T::from_usize(P).unwrap(); P],
                    };

                    pos.params_array
                        .column_iter_mut()
                        .enumerate()
                        .for_each(|(pdx, mut params)| params[pdx] += step_sizes[pdx]);
                    neg.params_array
                        .column_iter_mut()
                        .enumerate()
                        .for_each(|(pdx, mut params)| params[pdx] -= step_sizes[pdx]);

                    let mut pos_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), P);
                    let mut neg_output = DMatrix::<ObserVec<T, N>>::zeros(series.len(), P);

                    self.fsm_initialize_states_ensbl(series, &mut pos)?;
                    self.fsm_initialize_states_ensbl(series, &mut neg)?;

                    self.fsm_simulate_ensbl(
                        series,
                        &mut pos,
                        &mut pos_output.as_view_mut(),
                        None::<&mut ObserVecNoise<T, N>>,
                    )?;
                    self.fsm_simulate_ensbl(
                        series,
                        &mut neg,
                        &mut neg_output.as_view_mut(),
                        None::<&mut ObserVecNoise<T, N>>,
                    )?;

                    fim.row_iter_mut().enumerate().for_each(|(rdx, mut row)| {
                        row.iter_mut().enumerate().for_each(|(cdx, value)| {
                            if rdx <= cdx {
                                let dmu_a = pos_output
                                    .row_iter()
                                    .zip(neg_output.row_iter())
                                    .map(|(pos_col, neg_col)| {
                                        if step_sizes[rdx] == T::zero() {
                                            &pos_col[rdx] - &neg_col[rdx]
                                        } else {
                                            (&pos_col[rdx] - &neg_col[rdx])
                                                * (T!(0.5) / step_sizes[rdx])
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
                                                * (T!(0.5) / step_sizes[cdx])
                                        }
                                    })
                                    .collect::<Vec<ObserVec<T, N>>>();

                                // Normalization procedure.
                                // TODO: This is not required once proper cov-matrices exist.
                                let obs_norm_a =
                                    dmu_a.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);
                                let obs_norm_b =
                                    dmu_b.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);

                                let valid_indices =
                                    dmu_a.iter().map(|x| !x.any_nan()).collect::<Vec<bool>>();

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

                                let coviter = (0..valid_indices.len())
                                    .zip(series)
                                    .filter_map(|(i, scobs_i)| {
                                        if valid_indices[i] {
                                            Some((0..valid_indices.len()).zip(series).filter_map(
                                                |(j, scobs_j)| {
                                                    if valid_indices[j] {
                                                        Some(acfunc(
                                                            abs!(
                                                                *scobs_i.get_timestamp()
                                                                    - *scobs_j.get_timestamp()
                                                            ),
                                                            scobs_i.distance(&scobs_j),
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

                                let covariance = CovMatrix::from_matrix(
                                    &DMatrix::<T>::from_iterator(obs_norm_a, obs_norm_b, coviter)
                                        .as_view(),
                                )
                                .unwrap();

                                *value = (0..N)
                                    .map(|idx| {
                                        (dmu_a_mat.row(idx)
                                            * covariance.ref_inverse_matrix()
                                            * dmu_b_mat.row(idx).transpose())[(0, 0)]
                                    })
                                    .sum::<T>();
                            }
                        });
                    });

                    **fim += fim.transpose() - SMatrix::<T, P, P>::from_diagonal(&fim.diagonal());

                    Ok::<(), OcnusError<T>>(())
                })?;

                Ok::<(), OcnusError<T>>(())
            })?;

        Ok(results)
    }
}
