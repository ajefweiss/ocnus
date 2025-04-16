use chrono::Local;
use directories::UserDirs;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
use ocnus::{
    OcnusError,
    coords::XCState,
    forward::{
        ECHModel, FSMEnsbl, FisherInformation, OcnusFSM,
        filters::{
            ABCParticleFilter, ABCParticleFilterSettings, ParticleFilter, ParticleFilterError,
            ParticleFilterSettingsBuilder, SIRParticleFilter, root_mean_square_metric,
        },
    },
    math::CovMatrix,
    obser::{NoNoise, ObserVec, ObserVecNoise, ScObs, ScObsConf, ScObsSeries},
    prodef::{Constant1D, Uniform1D, UnivariateND},
};
use std::{io::prelude::*, path::Path};

fn main() {
    Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%dT%H:%M:%S.%f"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info)
        .init();

    let prior = UnivariateND::new([
        Uniform1D::new((-1.5, 1.5)).unwrap(),
        Uniform1D::new((1.5, 4.5)).unwrap(),
        Constant1D::new(0.0),
        Uniform1D::new((-0.5, 0.5)).unwrap(),
        Constant1D::new(1.0),
        Constant1D::new(0.15),
        Constant1D::new(0.75),
        Constant1D::new(375.0),
        Uniform1D::new((10.0, 25.0)).unwrap(),
        Constant1D::new(0.0),
        Constant1D::new(0.0),
        Uniform1D::new((5.0, 20.0)).unwrap(),
    ]);

    let model: ECHModel<f64, UnivariateND<f64, 12>> = ECHModel::new(prior);

    const SCOUNT: usize = 6;

    let sc1: ScObsSeries<f64, ObserVec<f64, 3>> =
        ScObsSeries::<_, ObserVec<_, 3>>::from_iterator((0..(SCOUNT)).map(|i| {
            ScObs::new(
                i as f64 * 6.0 * 3600.0 as f64 + 16.0 * 3600.0,
                ScObsConf::Position([1.0, 0.0, 0.0]),
                None,
            )
        }));

    let sc2: ScObsSeries<f64, ObserVec<f64, 3>> =
        ScObsSeries::<_, ObserVec<_, 3>>::from_iterator((0..(SCOUNT)).map(|i| {
            ScObs::new(
                i as f64 * 6.0 * 3600.0 as f64 + 16.0 * 3600.0,
                ScObsConf::Position([1.0, 0.0, 0.05]),
                None,
            )
        }));

    let mut sc = sc1 + sc2;
    sc.sort_by_timestamp();

    let mut fevmd_0 = FSMEnsbl {
        params_array: Matrix::<f64, _, Dyn, VecStorage<f64, Const<12>, Dyn>>::from_iterator(
            1,
            [
                -0.14, 3.18, 0.0, -0.1, 1.0, 0.15, 0.75, 375.0, 16.0, 0.0, 0.0, 11.5,
            ],
        ),
        fm_states: vec![(); 1],
        cs_states: vec![XCState::default(); 1],
        weights: vec![1.0],
    };

    const NOISE_VAR: f64 = 0.1;

    let auto_correlation = |dt, dd| {
        if dd > 0.0 {
            0.0
        } else if dt == 0.0 {
            NOISE_VAR
        } else {
            0.0
        }
    };

    let result = model
        .fischer_information_matrix(&sc, &fevmd_0, &auto_correlation)
        .unwrap();

    let fim = result[0];

    let fim_clean = fim;

    let fim_reduced = fim_clean
        .remove_row(10)
        .remove_column(10)
        .remove_row(9)
        .remove_column(9)
        .remove_row(7)
        .remove_column(7)
        .remove_row(6)
        .remove_column(6)
        .remove_row(5)
        .remove_column(5)
        .remove_row(4)
        .remove_column(4)
        .remove_row(2)
        .remove_column(2);

    println!("{:.5}", fim_clean);
    println!("{:.5}", fim_reduced);
    println!("{:.5}", fim_reduced.try_inverse().unwrap());

    // Create synthetic output.
    let mut output = DMatrix::<ObserVec<_, 3>>::zeros(sc.len(), 1);

    model
        .fsm_initialize_states_ensbl(&sc, &mut fevmd_0)
        .unwrap();
    model
        .fsm_simulate_ensbl(
            &sc,
            &mut fevmd_0,
            &mut output.as_view_mut(),
            None::<&mut NoNoise<f64>>,
        )
        .unwrap();

    let sc_synth =
        ScObsSeries::from_iterator(sc.into_iter().zip(output.row_iter()).map(|(mut obs, out)| {
            obs.set_observation(out[(0, 0)].clone());
            obs
        }));

    // let sc_synth = ScObsSeries::<_, ObserVec<_, 3>>::from_iterator((0..(SCOUNT)).map(|i| {
    //     ScObs::new(
    //         i as f64 * 3.0 * 3600.0 as f64 + 16.0 * 3600.0,
    //         ScObsConf::Distance(1.0),
    //         if output.row(i)[0].any_nan() {
    //             None
    //         } else {
    //             Some(output.row(i)[0].clone())
    //         },
    //     )
    // }));

    const ENSEMBLE_SIZE: usize = 2048;

    let mut pf_settings = ParticleFilterSettingsBuilder::default()
        .expl_factor(1.65)
        .quantiles([0.175, 0.5, 0.75])
        .rseed(70)
        .sim_time_limit(0.5 * SCOUNT as f64)
        .build()
        .unwrap();

    let init_result = model
        .pf_initialize_ensemble(
            &sc_synth,
            ENSEMBLE_SIZE,
            2_usize.pow(17),
            Some((&root_mean_square_metric, 10.0)),
            &mut pf_settings,
        )
        .unwrap();

    let base_dir_opt = if let Some(user_dirs) = UserDirs::new() {
        let doc_dir = user_dirs.document_dir().unwrap();

        let path = Path::new(doc_dir)
            .join("Data")
            .join("example_fisher_multi_ech");

        if path.exists() {
            Some(path)
        } else {
            warn!(
                "path {} not found, results will not be saved",
                path.into_os_string().into_string().unwrap()
            );

            None
        }
    } else {
        warn!("no document fold found, results will not be saved");

        None
    };

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("{}.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        init_result.save(format_str).unwrap();
    }

    let mut total_counter = 1;
    let mut fevmd = init_result.get_ensbl().clone();
    let mut epsilon = init_result.get_quantiles().unwrap()[0];
    let mut abort_counter = 0;
    let mut noise = ObserVecNoise::Gaussian(NOISE_VAR, 0);

    for _ in 1..10 {
        let mut abc_settings = ABCParticleFilterSettings::Threshold((
            pf_settings.clone(),
            &root_mean_square_metric,
            epsilon,
        ));

        let run = model.pf_abc_iter(
            &sc_synth,
            &fevmd,
            (ENSEMBLE_SIZE, 2_usize.pow(16)),
            &mut abc_settings,
            &mut noise,
        );

        let run_abc_result = match run {
            Ok(result) => {
                abort_counter = 0;
                result
            }
            Err(err) => match err {
                OcnusError::ParticleFilter(pferr) => match pferr {
                    ParticleFilterError::TimeLimitExceeded { .. } => break,
                    _ => {
                        abort_counter += 1;

                        if abort_counter < 3 {
                            abc_settings.rseed += 1;
                            continue;
                        } else {
                            break;
                        }
                    }
                },
                _ => panic!(),
            },
        };

        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("{}.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[total_counter]);

            run_abc_result.save(format_str).unwrap();
        }

        fevmd = run_abc_result.get_ensbl().clone();
        total_counter += 1;
        epsilon = run_abc_result.get_quantiles().unwrap()[0];
    }

    let iter_var = vec![
        100.0, 75.0, 50.0, 25.0, 15.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.75,
        0.5, 0.25, 0.15, 0.1, NOISE_VAR,
    ];

    for idx in 0..iter_var.len() {
        let mut sir_settings = ParticleFilterSettingsBuilder::default()
            .expl_factor(1.0)
            .build()
            .unwrap();

        let mut noise = ObserVecNoise::Multivariate(
            CovMatrix::from_matrix(
                &DMatrix::from_diagonal_element(sc_synth.len(), sc_synth.len(), iter_var[idx])
                    .as_view(),
            )
            .unwrap(),
            idx as u64 * 13,
        );

        let run = model.pf_sir_iter(
            &sc_synth,
            &fevmd,
            (ENSEMBLE_SIZE, 4 * ENSEMBLE_SIZE),
            &mut sir_settings,
            &mut noise,
        );

        let run_sir_result = run.unwrap();

        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("{}.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[total_counter]);

            run_sir_result.save(format_str).unwrap();
        }

        fevmd = run_sir_result.get_ensbl().clone();
        total_counter += 1;
    }
}
