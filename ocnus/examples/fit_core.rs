use chrono::Local;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::DMatrix;
use ocnus::{
    OcnusError,
    forward::{
        COREModel,
        filters::{
            ABCParticleFilter, ABCParticleFilterSettings, ParticleFilter, ParticleFilterError,
            ParticleFilterSettingsBuilder, SIRParticleFilter, root_mean_square_metric,
        },
    },
    math::CovMatrix,
    obser::{ObserVec, ObserVecNoise, ScObs, ScObsConf, ScObsSeries},
    prodef::{Constant1D, Uniform1D, UnivariateND},
};
use std::{fs::create_dir, io::prelude::*, path::Path};

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
        Uniform1D::new((-1.0, 1.0)).unwrap(),
        Uniform1D::new((-0.25, 0.25)).unwrap(),
        Uniform1D::new((-0.5, 0.5)).unwrap(),
        Constant1D::new(20.0),
        Uniform1D::new((0.1, 0.35)).unwrap(),
        Uniform1D::new((0.25, 1.0)).unwrap(),
        Uniform1D::new((600.0, 1500.0)).unwrap(),
        Uniform1D::new((10.0, 30.0)).unwrap(),
        Uniform1D::new((-50.0, 10.0)).unwrap(),
        Constant1D::new(400.0),
        Constant1D::new(1.0),
    ]);

    let model = COREModel::new(prior);

    #[allow(clippy::excessive_precision)]
    let refobs = [
        // ObserVec::from([-2.67159853, -13.38264243, -7.20006469]),
        ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        // ObserVec::from([-5.34451816, -14.85328723, -1.73480069]),
        ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        // ObserVec::from([-4.00239361, -14.94369, 1.2604304]),
        ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        // ObserVec::from([-2.7273795, -14.29364075, 4.8267508]),
        ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        // ObserVec::from([-4.28371012, -13.89409455, 5.13879915]),
        ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    const ENSEMBLE_SIZE: usize = 2048;

    let sc = ScObsSeries::<f64, ObserVec<f64, 3>>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            3.5 * 24.0 * 3600.0 + i as f64 * 3600.0 * 2.0,
            ScObsConf::Distance(1.0),
            Some(refobs[i].clone()),
        )
    }));

    let mut pf_settings = ParticleFilterSettingsBuilder::default()
        .expl_factor(2.0)
        .quantiles([0.15, 0.5, 0.8])
        .rseed(70)
        .sim_time_limit(2.5)
        .build()
        .unwrap();

    let init_result = model
        .pf_initialize_ensemble(
            &sc,
            ENSEMBLE_SIZE,
            2_usize.pow(17),
            Some((&root_mean_square_metric, 9.0)),
            &mut pf_settings,
        )
        .unwrap();

    let path = Path::new("ocnus")
        .join("examples")
        .join("output")
        .join("fit_core");

    let base_dir_opt = if path.exists() {
        Some(path)
    } else {
        if path.parent().unwrap().parent().unwrap().exists() {
            create_dir(&path).expect("failed to create output directory");

            Some(path)
        } else {
            warn!(
                "path {} not found, results will not be saved",
                path.into_os_string().into_string().unwrap()
            );

            None
        }
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
    let mut noise = ObserVecNoise::Gaussian(1.0, 0);

    for _ in 1..20 {
        let mut abc_settings = ABCParticleFilterSettings::Threshold((
            pf_settings.clone(),
            &root_mean_square_metric,
            epsilon,
        ));

        let run = model.pf_abc_iter(
            &sc,
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

    for idx in 0..10 {
        let mut sir_settings = ParticleFilterSettingsBuilder::default()
            .expl_factor(1.0)
            .build()
            .unwrap();

        let mut noise = ObserVecNoise::Multivariate(
            CovMatrix::from_matrix(
                &DMatrix::from_diagonal_element(
                    sc.len(),
                    sc.len(),
                    1.0 + (10.0 / (1 + idx) as f64),
                )
                .as_view(),
            )
            .unwrap(),
            idx * 13,
        );

        let run = model.pf_sir_iter(
            &sc,
            &fevmd,
            (ENSEMBLE_SIZE, 2_usize.pow(16)),
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
