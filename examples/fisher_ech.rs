use chrono::Local;
use core::panic;
use covmatrix::CovMatrix;
use dyn_fmt::AsStrFormatExt;
use env_logger::Builder;
use log::warn;
use nalgebra::{DMatrix, DVector, DVectorView, SVector, U1, U12};
use ocnus::{
    base::{OcnusEnsbl, OcnusModel, ScObs, ScObsConf, ScObsSeries},
    coords::XCState,
    methods::{
        filters::{ParticleFilterBuilder, ParticleFilterSettingsBuilder},
        fisher_information_matrix,
    },
    models::ECHModel,
    obser::{
        MeasureInSituMagneticFields, NullNoise, ObserVec, ObserVecNoise,
        multivariate_log_likelihood, observec_rmsep,
    },
    stats::{ConstantDensity, Density, MultivariateDensity, UniformDensity},
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

    let prior = MultivariateDensity::<f64, 12>::new(&[
        // UniformDensity::new((-1.5, 1.5)).unwrap(),
        // UniformDensity::new((1.5, 4.5)).unwrap(),
        ConstantDensity::new(-0.22),
        ConstantDensity::new(3.06),
        ConstantDensity::new(0.0),
        ConstantDensity::new(-0.11),
        ConstantDensity::new(1.0),
        ConstantDensity::new(0.11),
        ConstantDensity::new(0.87),
        ConstantDensity::new(500.0),
        UniformDensity::new((10.0, 25.0)).unwrap(),
        ConstantDensity::new(0.0),
        ConstantDensity::new(0.0),
        UniformDensity::new((0.0, 10.0)).unwrap(),
    ]);

    let x_0 = SVector::from([
        -0.22, 3.06, 0.0, -0.11, 1.0, 0.11, 0.87, 500.0, 15.4, 0.0, 0.0, 5.2,
    ]);

    assert!((&prior).validate_sample(&x_0.as_view()));

    let range = (&prior).get_range();
    let model = ECHModel::<f64, _>::new(prior);

    const NOISE_STD: f64 = 0.15;
    const SCOUNT: usize = 10;

    // CovMatrix only for non-NaN observations.
    let lh_covm = CovMatrix::new(
        DMatrix::from_diagonal_element(SCOUNT, SCOUNT, NOISE_STD.powi(2)),
        true,
    )
    .unwrap();

    let lh_covm_large = CovMatrix::new(
        DMatrix::from_diagonal_element(SCOUNT, SCOUNT, 2.0 * NOISE_STD.powi(2)),
        true,
    )
    .unwrap();

    // SC is set up so the first/last observations are near the boundary.
    let sc_fim = ScObsSeries::<f64>::from_iterator((0..SCOUNT).map(|i| {
        ScObs::new(
            i as f64 * 3600.0 * (16.0 / (SCOUNT - 1) as f64) + 14400.0,
            ScObsConf::Position([1.0, 0.0, 0.0]),
        )
    }));

    let mut sc = ScObsSeries::new();

    sc += ScObs::new(0.0, ScObsConf::Position([1.0, 0.0, 0.0]));

    for v in (&sc_fim).into_iter() {
        sc += v.clone();
    }

    sc += ScObs::new(86400.0, ScObsConf::Position([1.0, 0.0, 0.0]));

    // Create synthetic output.
    let mut ensbl = OcnusEnsbl::new(1, range);

    ensbl
        .ptpdf
        .particles_mut()
        .iter_mut()
        .zip(x_0.iter())
        .for_each(|(v1, v2)| *v1 = *v2);

    let mut output_diag = DMatrix::<ObserVec<_, 12>>::zeros(sc_fim.len(), 1);

    model.initialize_states_ensbl(&mut ensbl).unwrap();
    model
        .simulate_ics_basis_ensbl(&sc_fim, &mut ensbl, &mut output_diag.as_view_mut())
        .unwrap();

    assert!(
        (output_diag[(0, 0)][0] > 0.65) && (output_diag[(0, 0)][0] < 1.0),
        "first observation is invalid"
    );
    assert!(
        (output_diag[(SCOUNT - 1, 0)][0] > 0.95) && (output_diag[(SCOUNT - 1, 0)][0] < 1.0),
        "last observation is invalid"
    );

    let result = fisher_information_matrix(
        &model,
        &sc_fim,
        &x_0.as_view::<U12, U1, U1, U12>(),
        &ECHModel::observe_mag3,
        &lh_covm,
    )
    .expect("fim calculation failed");

    let fim_reduced = result
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
        .remove_row(3)
        .remove_column(3)
        .remove_row(2)
        .remove_column(2)
        .remove_row(1)
        .remove_column(1)
        .remove_row(0)
        .remove_column(0);

    println!("Reduced FIM = {:.2}", fim_reduced);
    println!(
        "Inverted Reduced FIM = {:.4}",
        fim_reduced.pseudo_inverse(1e-6).unwrap()
    );

    let mut output = DMatrix::<ObserVec<_, 3>>::zeros(sc.len(), 1);

    model.initialize_states_ensbl(&mut ensbl).unwrap();
    model
        .simulate_ensbl(
            &sc,
            &mut ensbl,
            &ECHModel::observe_mag3,
            &mut output.as_view_mut(),
            None::<&mut NullNoise<f64>>,
        )
        .unwrap();

    let refobs = Vec::from_iter(output.iter().cloned());

    const ENSEMBLE_SIZE: usize = 4096;

    let sc_obser = DVector::from_iterator(sc.len(), (0..sc.len()).map(|idx| refobs[idx].clone()));

    let err_func = |o: &DVectorView<ObserVec<f64, 3>>| observec_rmsep(o, &sc_obser.as_view());

    let llh_func = |o: &DVectorView<ObserVec<f64, 3>>| {
        multivariate_log_likelihood(o, &sc_obser.as_view(), &lh_covm)
    };

    let llh_func_large = |o: &DVectorView<ObserVec<f64, 3>>| {
        multivariate_log_likelihood(o, &sc_obser.as_view(), &lh_covm_large)
    };

    let mut pf = ParticleFilterBuilder::<
        ECHModel<f64, _>,
        f64,
        12,
        (),
        XCState<f64>,
        ObserVec<f64, 3>,
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

    pf.pf_initialize_ensbl(&pfs, &ECHModel::observe_mag3, (&err_func, 0.8))
        .unwrap();

    let path = Path::new("examples").join("output").join("fisher_ech");

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

    let mut noise = ObserVecNoise::Gaussian(NOISE_STD, 0);

    pfs.max_iterations = 2;

    let abc_loop_result = pf.pf_abc_loop(&pfs, &mut noise, &ECHModel::observe_mag3, &err_func);

    if abc_loop_result.is_ok() {
        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("abc_1.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[0]);

            pf.save(format_str).unwrap();
        }
    } else {
        panic!()
    }

    let abc_loop_result = pf.pf_abc_loop(&pfs, &mut noise, &ECHModel::observe_mag3, &err_func);

    pf.ensbl.as_mut().unwrap().ptpdf.update_mvpdf();

    println!(
        "\nABC STD_THETA = {:.5}",
        &DVector::from_iterator(
            fim_reduced.nrows(),
            pf.ensbl
                .as_ref()
                .unwrap()
                .ptpdf
                .covmatrix()
                .matrix()
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
                .remove_row(3)
                .remove_column(3)
                .remove_row(2)
                .remove_column(2)
                .remove_row(1)
                .remove_column(1)
                .remove_row(0)
                .remove_column(0)
                .diagonal()
                .iter()
                .map(|v| v.sqrt())
        )
        .transpose()
    );

    if abc_loop_result.is_ok() {
        if let Some(base_dir) = &base_dir_opt {
            let format_str = base_dir
                .join("abc_2.particles")
                .into_os_string()
                .into_string()
                .unwrap()
                .format(&[0]);

            pf.save(format_str).unwrap();
        }
    } else {
        panic!()
    }

    pfs.expl_factor = 1.0;
    pfs.max_iterations = 3;

    let sir_loop_result = pf.pf_sir_loop(&pfs, &ECHModel::observe_mag3, &llh_func_large);

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

    pfs.max_iterations = 3;

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

    println!(
        "\nFIM STD_THETA = {:.5}",
        &DVector::from_iterator(
            fim_reduced.nrows(),
            fim_reduced
                .pseudo_inverse(1e-6)
                .unwrap()
                .diagonal()
                .iter()
                .map(|v| v.sqrt())
        )
        .transpose()
    );

    println!(
        "\nSIR STD_THETA = {:.5}",
        &DVector::from_iterator(
            fim_reduced.nrows(),
            pf.ensbl
                .as_ref()
                .unwrap()
                .ptpdf
                .covmatrix()
                .matrix()
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
                .remove_row(3)
                .remove_column(3)
                .remove_row(2)
                .remove_column(2)
                .remove_row(1)
                .remove_column(1)
                .remove_row(0)
                .remove_column(0)
                .diagonal()
                .iter()
                .map(|v| v.sqrt())
        )
        .transpose()
    );
}
