use crate::fevm::filters::ParticleFilter;
use serde::{Deserialize, Serialize};

/// A trait that enables the calculation of the Fisher Information Matrix (FIM) for a [`ForwardEnsembleVectorModel`].
pub trait FisherInformation<const P: usize, const N: usize, FS, GS>:
    ParticleFilter<P, N, FS, GS>
where
    Self: Sync,
    FS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
    GS: Clone + std::fmt::Debug + Default + for<'a> Deserialize<'a> + Send + Serialize,
{
    /// Compute the Fisher information matrix (FIM) for an array of model parameters.
    fn fischer_information_matrix(
        &self,
        scobs: &ScObs,
        ensemble: &mut EnsblData<S, P>,
        covm_inv: DMatrixView<Fp>,
    ) -> Vec<SMatrix<Fp, P, P>> {
        let step_sizes = self.adaptive_step_sizes();

        let mut result = vec![SMatrix::<Fp, P, P>::zeros(); ensemble.len()];

        ensemble
            .par_chunks_mut(Self::RCS / 4)
            .zip(result.par_chunks_mut(Self::RCS / 4))
            .for_each(|(data_chunks, result_chunks)| {
                data_chunks.iter_mut().zip(result_chunks.iter_mut()).for_each(|(data, fim)| {
                    let mut dap = EnsblData {
                        is_initialized: false,
                        data: (0..P).map(|_| data.clone()).collect(),
                    };

                    let mut dam = EnsblData {
                        is_initialized: false,
                        data: (0..P).map(|_| data.clone()).collect(),
                    };

                    dap.iter_mut()
                        .enumerate()
                        .for_each(|(pdx, dap_data)| dap_data.params[pdx] += step_sizes[pdx]);

                    dam.iter_mut()
                        .enumerate()
                        .for_each(|(pdx, dam_data)| dam_data.params[pdx] -= step_sizes[pdx]);

                    let mut dap_output = ndarray::Array2::<Option<ModelObserVec<N>>>::default((scobs.len(), P));
                    let mut dam_output = ndarray::Array2::<Option<ModelObserVec<N>>>::default((scobs.len(), P));

                    self.initialize_states_ensemble(scobs, &mut dap);
                    self.initialize_states_ensemble(scobs, &mut dam);

                    self.forward_simulation(scobs, &mut dap, &mut dap_output.view_mut(), false);
                    self.forward_simulation(scobs, &mut dam, &mut dam_output.view_mut(), false);

                    fim.row_iter_mut().enumerate().for_each(|(rdx, mut row)| {
                        row.iter_mut().enumerate().for_each(|(cdx, value)| {
                            if rdx <= cdx {
                                let dmu_a = dap_output
                                    .axis_iter(Axis(0))
                                    .zip(dam_output.axis_iter(Axis(0)))
                                    .map(|(dap, dam)| {
                                        dap[rdx]
                                            .as_ref()
                                            .zip(dam[rdx].as_ref())
                                            .map(|(v_p, v_m)| Some((v_p - v_m) * (0.5 / step_sizes[rdx])))
                                            .unwrap_or(None)
                                    })
                                    .collect::<Vec<Option<ModelObserVec<N>>>>();

                                let dmu_b = dap_output
                                    .axis_iter(Axis(0))
                                    .zip(dam_output.axis_iter(Axis(0)))
                                    .map(|(dap, dam)| {
                                        dap[cdx]
                                            .as_ref()
                                            .zip(dam[cdx].as_ref())
                                            .map(|(v_p, v_m)| Some((v_p - v_m) * (0.5 / step_sizes[cdx])))
                                            .unwrap_or(None)
                                    })
                                    .collect::<Vec<Option<ModelObserVec<N>>>>();

                                // Normalization procedure.
                                // TODO: This is not required once proper cov-matrices exist.
                                let obs_norm_a = dmu_a.iter().fold(0, |acc, x| acc + x.is_some() as usize);
                                let obs_norm_b = dmu_b.iter().fold(0, |acc, x| acc + x.is_some() as usize);

                                assert!(
                                    obs_norm_a == obs_norm_b,
                                    "fim evaluated at an invalid location due to differing lengths of observations"
                                );

                                let dmu_a_mat = DMatrix::from_iterator(
                                    N,
                                    obs_norm_a,
                                    dmu_a.iter().filter_map(|v| v.as_ref().map(|x| x.iter().copied())).flatten(),
                                );

                                let dmu_b_mat = DMatrix::from_iterator(
                                    N,
                                    obs_norm_b,
                                    dmu_b.iter().filter_map(|v| v.as_ref().map(|x| x.iter().copied())).flatten(),
                                );

                                *value = (0..N)
                                    .map(|idx| (dmu_a_mat.row(idx) * covm_inv * dmu_b_mat.row(idx).transpose())[(0, 0)])
                                    .sum::<Fp>();
                            }
                        });
                    });

                    *fim += fim.transpose() - SMatrix::<Fp, P, P>::from_diagonal(&fim.diagonal());
                });
            });

        result
    }
}
