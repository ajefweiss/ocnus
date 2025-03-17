use crate::fevm::FEVMData;
use crate::stats::CovMatrix;
use crate::{ScObsSeries, fevm::filters::ParticleFilter, obser::ObserVec};
use nalgebra::{Const, DMatrix, Dyn, Matrix, SMatrix, VecStorage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::FEVMError;
use super::noise::FEVMNoiseNull;

/// A trait that enables the calculation of the Fisher Information Matrix (FIM) for a [`ForwardEnsembleVectorModel`].
pub trait FisherInformation<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    Self: Sync,
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
{
    /// Compute the Fisher information matrix (FIM) for an array of model parameters.
    fn fischer_information_matrix<F>(
        &self,
        series: &ScObsSeries<ObserVec<N>>,
        fevmd: &FEVMData<P, FS, GS>,
        corrfunc: &F,
    ) -> Result<Vec<SMatrix<f32, P, P>>, FEVMError> 
      where F:Fn(f32) -> f32 + Sync {
        let step_sizes = self.param_step_sizes();

        let mut results = vec![SMatrix::<f32, P, P>::zeros(); fevmd.params.ncols()];

        fevmd
            .params
            .par_column_iter()
            .zip(results.par_iter_mut())
            .chunks(Self::RCS / 4)
            .try_for_each(|mut chunks| {
               
                chunks.iter_mut().try_for_each(|(params_ref, fim)| {
                    let mut dap = FEVMData {
                        params: Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::from_columns(&[*params_ref; P]),
                        fevm_states: vec![FS::default(); P],
                        geom_states: vec![GS::default(); P],
                        weights: vec![1.0 / P as f32; P],
                    };

                    let mut dam = FEVMData {
                        params: Matrix::<f32, Const<P>, Dyn, VecStorage<f32, Const<P>, Dyn>>::from_columns(&[*params_ref; P]),
                        fevm_states: vec![FS::default(); P],
                        geom_states: vec![GS::default(); P],
                        weights: vec![1.0 / P as f32; P],
                    };

                    dap.params
                        .column_iter_mut()
                        .enumerate()
                        .for_each(|(pdx, mut params)| {
                            params[pdx] +=  step_sizes[pdx]
                        });
                    dam.params
                        .column_iter_mut()
                        .enumerate()
                        .for_each(|(pdx, mut params)| {
                            params[pdx] -=  step_sizes[pdx]
                        });

                    let mut dap_output = DMatrix::<ObserVec<N>>::zeros(series.len(), P);
                    let mut dam_output = DMatrix::<ObserVec<N>>::zeros(series.len(), P);

                    self.fevm_initialize_states_only(series, &mut dap)?;
                    self.fevm_initialize_states_only(series, &mut dam)?;

                    self.fevm_simulate(
                        series,
                        &mut dap,
                        &mut dap_output,
                        None::<(&FEVMNoiseNull, u64)>,
                    )?;

                    self.fevm_simulate(
                        series,
                        &mut dam,
                        &mut dam_output,
                        None::<(&FEVMNoiseNull, u64)>,
                    )?;


                    fim.row_iter_mut().enumerate().for_each(|(rdx, mut row)| {
                        row.iter_mut().enumerate().for_each(|(cdx, value)| {
                            if rdx <= cdx {
                                
                                let dmu_a = dap_output.row_iter().zip(dam_output.row_iter()).map(|(dap_col, dam_col)| {
                                  (&dap_col[rdx] - &dam_col[rdx]) * (0.5 / step_sizes[rdx]) 
                                  
                                }).collect::<Vec<ObserVec<N>>>();

                                

                                let dmu_b = dap_output.row_iter().zip(dam_output.row_iter()).map(|(dap_col, dam_col)| {
                                    (&dap_col[cdx] - &dam_col[cdx]) * (0.5 / step_sizes[cdx]) 
                                }).collect::<Vec<ObserVec<N>>>();
                  
                                // Normalization procedure.
                                // TODO: This is not required once proper cov-matrices exist.
                                let obs_norm_a = dmu_a.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);
                                let obs_norm_b  = dmu_b.iter().fold(0, |acc, x| acc + !x.any_nan() as usize);

                                assert!(
                                    obs_norm_a == obs_norm_b,
                                    "fim evaluated at an invalid location due to differing lengths of observations"
                                );

                                assert!(obs_norm_a > 0, "no valid evaluation points");

                                let valid_indices = dmu_a.iter().map(|x| !x.any_nan()).collect::<Vec<bool>>();

                                let dmu_a_mat = DMatrix::from_iterator(
                                    N,
                                    obs_norm_a,
                                    dmu_a.iter().filter_map(|obsvec| if !obsvec.any_nan() {Some(obsvec.0.iter().copied())} else {None}).flatten(),
                                );
                     
                                let dmu_b_mat = DMatrix::from_iterator(
                                    N,
                                    obs_norm_b,
                                    dmu_b.iter().filter_map(|obsvec| if !obsvec.any_nan() {Some(obsvec.0.iter().copied())} else {None}).flatten(),
                                );

                                let coviter = (0..valid_indices.len()).zip(series).filter_map(| (i,scobs_i)| if valid_indices[i] {Some(
                                (0..valid_indices.len()).zip(series).filter_map(|(j, scobs_j)| if valid_indices[j] {Some(corrfunc((scobs_i.timestamp() - scobs_j.timestamp()).abs()))} else {None} ))} else {None}).flatten();
                              
                         
                                let covariance = CovMatrix::from_matrix(&DMatrix::<f32>::from_iterator(obs_norm_a, obs_norm_b,coviter).as_view()).unwrap();
                             
                                *value = (0..N)
                                    .map(|idx| (dmu_a_mat.row(idx) * covariance.inverse_matrix() * dmu_b_mat.row(idx).transpose())[(0, 0)])
                                    .sum::<f32>();
                            }
                        });
                    });

                    **fim += fim.transpose() - SMatrix::<f32, P, P>::from_diagonal(&fim.diagonal());

                    Ok::<(), FEVMError>(())
                })?;

                Ok::<(), FEVMError>(())
            })?;

        Ok(results)
    }
}
