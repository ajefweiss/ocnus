use chrono::Local;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::{Const, OVector};
use ocnus::{
    base::{ScObs, ScObsConf, ScObsSeries},
    methods::filters::ParticleFilterBuilder,
    models::ECHModel,
    obser::{ObserVec, observec_rmsep},
    stats::{ConstantDensity, MultivariateDensity, ReciprocalDensity, UniformDensity},
};
use std::{fs::create_dir_all, io::prelude::*, path::Path};

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

    let prior = MultivariateDensity::<f32, Const<12>>::new(&[
        UniformDensity::new((-1.5, 1.5)).unwrap(),
        UniformDensity::new((1.5, 4.5)).unwrap(),
        ConstantDensity::new(1.0),
        UniformDensity::new((-0.75, 0.75)).unwrap(),
        ConstantDensity::new(1.0),
        ReciprocalDensity::new((0.05, 0.25)).unwrap(),
        UniformDensity::new((0.85, 1.0)).unwrap(),
        ConstantDensity::new(375.0),
        UniformDensity::new((10.0, 25.0)).unwrap(),
        ConstantDensity::new(0.0),
        ConstantDensity::new(0.0),
        UniformDensity::new((0.0, 30.0)).unwrap(),
    ]);

    let model = ECHModel::new(prior);

    #[allow(clippy::excessive_precision)]
    let refobs = OVector::from([
        ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ]);

    const ENSEMBLE_SIZE: usize = 2048;

    let sc = ScObsSeries::<f32>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            10800.0 + i as f32 * 3600.0 * 2.0,
            ScObsConf::Position([1.0, 0.0, 0.0]),
        )
    }));

    // Does nothng, but a good test.
    let refobs_sorted = sc.sort_data_by_timestamp(&[&refobs.data.0]);

    let mut pfobj = ParticleFilterBuilder::default()
        .expl_factor(2.0)
        .rseed(70)
        .sim_time_limit(2.5)
        .build()
        .unwrap();

    let ef_closure = |view| observec_rmsep(view, &refobs.as_view());

    pfobj
        .pf_initialize_ensbl(
            &model,
            &sc,
            (ENSEMBLE_SIZE, 2_usize.pow(17)),
            &ECHModel::<f32, MultivariateDensity<f32, Const<12>>>::observe_mag,
            (&ef_closure, 1.0),
        )
        .unwrap();

    let path = Path::new("ocnus")
        .join("examples")
        .join("output")
        .join("fit_ech");

    let base_dir_opt = if path.exists() {
        Some(path)
    } else if path.parent().unwrap().parent().unwrap().exists() {
        create_dir_all(&path).expect("failed to create output directory");

        Some(path)
    } else {
        warn!(
            "path {} not found, results will not be saved",
            path.into_os_string().into_string().unwrap()
        );

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
    let mut noise = ObserVecNoise::Gaussian(1.0, 0);

    // for _ in 1..20 {
    //     let mut abc_settings =
    //         ABCSettings::Threshold((pf_settings.clone(), &observec_rmse, epsilon));

    //     let run = model.pf_abc_iter(
    //         &sc,
    //         &fevmd,
    //         (ENSEMBLE_SIZE, 2_usize.pow(16)),
    //         &mut abc_settings,
    //         &mut noise,
    //     );

    //     let run_abc_result = match run {
    //         Ok(result) => {
    //             abort_counter = 0;
    //             result
    //         }
    //         Err(err) => match err {
    //             OcnusError::FilterError(pferr) => match pferr {
    //                 ParticleFilterError::TimeLimitExceeded { .. } => break,
    //                 _ => {
    //                     abort_counter += 1;

    //                     if abort_counter < 3 {
    //                         abc_settings.rseed += 1;
    //                         continue;
    //                     } else {
    //                         break;
    //                     }
    //                 }
    //             },
    //             _ => panic!(),
    //         },
    //     };

    //     if let Some(base_dir) = &base_dir_opt {
    //         let format_str = base_dir
    //             .join("{}.particles")
    //             .into_os_string()
    //             .into_string()
    //             .unwrap()
    //             .format(&[total_counter]);

    //         run_abc_result.save(format_str).unwrap();
    //     }

    //     fevmd = run_abc_result.get_ensbl().clone();
    //     total_counter += 1;
    //     epsilon = run_abc_result.get_quantiles().unwrap()[0];
    // }

    // for idx in 0..10 {
    //     let mut sir_settings = ParticleFilterSettingsBuilder::default()
    //         .expl_factor(1.0)
    //         .build()
    //         .unwrap();

    //     let mut noise = ObserVecNoise::Multivariate(
    //         CovMatrix::from_matrix(
    //             &DMatrix::from_diagonal_element(
    //                 sc.len(),
    //                 sc.len(),
    //                 1.0 + (10.0 / (1 + idx) as f32),
    //             )
    //             .as_view(),
    //         )
    //         .unwrap(),
    //         idx * 13,
    //     );

    //     let run = model.pf_sir_iter(
    //         &sc,
    //         &fevmd,
    //         (ENSEMBLE_SIZE, 2_usize.pow(16)),
    //         &mut sir_settings,
    //         &mut noise,
    //     );

    //     let run_sir_result = run.unwrap();

    //     if let Some(base_dir) = &base_dir_opt {
    //         let format_str = base_dir
    //             .join("{}.particles")
    //             .into_os_string()
    //             .into_string()
    //             .unwrap()
    //             .format(&[total_counter]);

    //         run_sir_result.save(format_str).unwrap();
    //     }

    //     fevmd = run_sir_result.get_ensbl().clone();
    //     total_counter += 1;
    // }
}
