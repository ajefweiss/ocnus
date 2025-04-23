use chrono::Local;
use covmatrix::CovMatrix;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::{DMatrix, DVector, DVectorView};
use ocnus::{
    base::{ScObs, ScObsConf, ScObsSeries},
    coords::XCState,
    methods::filters::{ParticleFilterBuilder, ParticleFilterSettingsBuilder},
    models::ECHModel,
    obser::{MeasureInSituMagneticFields, ObserVec, multivariate_log_likelihood, observec_rmsep},
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

    let prior = MultivariateDensity::new(&[
        UniformDensity::new((-1.5, 1.5)).unwrap(),
        UniformDensity::new((1.5, 4.5)).unwrap(),
        ConstantDensity::new(0.0),
        UniformDensity::new((-0.75, 0.75)).unwrap(),
        ConstantDensity::new(1.0),
        ReciprocalDensity::new((0.05, 0.25)).unwrap(),
        UniformDensity::new((0.25, 1.0)).unwrap(),
        ConstantDensity::new(450.0),
        UniformDensity::new((10.0, 25.0)).unwrap(),
        ConstantDensity::new(0.0),
        ConstantDensity::new(0.0),
        UniformDensity::new((0.0, 20.0)).unwrap(),
    ]);

    let model = ECHModel::new(prior);

    #[allow(clippy::excessive_precision)]
    let refobs = [
        ObserVec::default(),
        ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
        ObserVec::default(),
    ];

    const ENSEMBLE_SIZE: usize = 2 * 2048;

    let sc = ScObsSeries::<f32>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            i as f32 * 3600.0 * 4.0,
            ScObsConf::Position([1.0, 0.0, 0.0]),
        )
    }));

    let sc_obser = DVector::from_iterator(sc.len(), (0..sc.len()).map(|idx| refobs[idx].clone()));

    // CovMatrix only for non-NaN observations.
    let lh_covm = CovMatrix::new(DMatrix::from_diagonal_element(5, 5, 1.0), true).unwrap();

    let err_func = |o: &DVectorView<ObserVec<f32, 3>>| observec_rmsep(o, &sc_obser.as_view());
    let llh_func = |o: &DVectorView<ObserVec<f32, 3>>| {
        multivariate_log_likelihood(o, &sc_obser.as_view(), &lh_covm)
    };

    let mut pf = ParticleFilterBuilder::<
        ECHModel<f32, _>,
        f32,
        12,
        (),
        XCState<f32>,
        ObserVec<f32, 3>,
    >::default()
    .model(model)
    .rseed(43)
    .build()
    .unwrap();

    let mut pfs = ParticleFilterSettingsBuilder::default()
        .series(sc)
        .ensemble_size(ENSEMBLE_SIZE)
        .simulation_ensemble_size(4 * ENSEMBLE_SIZE)
        .expl_factor(2.0)
        .simulation_time_limit(10.0)
        .error_quantile(0.2)
        .build()
        .unwrap();

    pf.pf_initialize_ensbl(&pfs, &ECHModel::observe_mag3, (&err_func, 0.65))
        .unwrap();

    let path = Path::new("examples").join("output").join("fit_ech_dev_sir");

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
            .join("initial.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }

    pfs.max_iterations = 500;

    let _dev_loop_result = pf.diff_ev_loop(&pfs, (0.15, 0.9), &ECHModel::observe_mag3, &err_func);

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("dev_1.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }

    let _dev_loop_result = pf.diff_ev_loop(&pfs, (0.05, 0.925), &ECHModel::observe_mag3, &err_func);

    if let Some(base_dir) = &base_dir_opt {
        let format_str = base_dir
            .join("dev_2.particles")
            .into_os_string()
            .into_string()
            .unwrap()
            .format(&[0]);

        pf.save(format_str).unwrap();
    }

    pfs.max_iterations = 5;

    let sir_loop_result = pf.pf_sir_loop(&pfs, &ECHModel::observe_mag3, &llh_func);

    if sir_loop_result.is_ok() {
        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("sir_1.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[0]);

            pf.save(format_str).unwrap();
        }
    } else {
        panic!()
    }

    let sir_loop_result = pf.pf_sir_loop(&pfs, &ECHModel::observe_mag3, &llh_func);

    if sir_loop_result.is_ok() {
        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("sir_2.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[0]);

            pf.save(format_str).unwrap();
        }
    } else {
        panic!()
    }
}
