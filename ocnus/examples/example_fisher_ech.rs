use chrono::Local;
use core::f64;
use env_logger::Builder;
use log::info;
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
use ocnus::{
    fevm::{
        ECHModel, FEVM, FEVMData, FEVMError, FEVMNoise, FEVMNullState, FisherInformation,
        filters::{
            ABCParticleFilter, ABCParticleFilterMode, ParticleFilter, ParticleFilterError,
            ParticleFilterSettingsBuilder, SIRParticleFilter, root_mean_square_filter,
        },
    },
    geom::XCState,
    obser::{ObserVec, ScObs, ScObsConf, ScObsSeries},
    stats::{CovMatrix, PDFConstant, PDFReciprocal, PDFUniform, PDFUnivariates},
};
use std::io::prelude::*;

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

    let prior = PDFUnivariates::new([
        PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
        PDFUniform::new_uvpdf((1.5, 4.5)).unwrap(),
        PDFConstant::new_uvpdf(0.0),
        PDFUniform::new_uvpdf((-0.5, 0.5)).unwrap(),
        PDFConstant::new_uvpdf(1.0),
        PDFReciprocal::new_uvpdf((0.1, 0.3)).unwrap(),
        PDFUniform::new_uvpdf((0.7, 0.9)).unwrap(),
        PDFConstant::new_uvpdf(1125.0),
        PDFUniform::new_uvpdf((10.0, 25.0)).unwrap(),
        PDFConstant::new_uvpdf(1.0),
        PDFUniform::new_uvpdf((3.0, 12.0)).unwrap(),
        PDFConstant::new_uvpdf(0.0),
    ]);

    let model = ECHModel::new(prior);

    let sc: ScObsSeries<f64, ObserVec<f64, 3>> = ScObsSeries::<_, ObserVec<_, 3>>::from_iterator(
        (0..20).map(|i| ScObs::new(i as f64 * 1.0 * 3600.0, ScObsConf::Distance(1.0), None)),
    );

    let mut fevmd_0 = FEVMData {
        params: Matrix::<f64, _, Dyn, VecStorage<f64, Const<12>, Dyn>>::from_iterator(
            1,
            [
                -15_f64.to_radians(),
                177.20819_f64.to_radians(),
                0.0_f64.to_radians(),
                -0.04,
                1.0,
                0.2,
                0.75,
                1125.0,
                16.0,
                1.0,
                6.0,
                0.0,
            ],
        ),
        fevm_states: vec![FEVMNullState::default(); 1],
        geom_states: vec![XCState::default(); 1],
        weights: vec![1.0],
    };

    const NOISE_VAR: f64 = 1.0;

    let corrfunc = |d| if d == 0.0 { NOISE_VAR } else { 0.0 };

    let result = model
        .fischer_information_matrix(&sc, &fevmd_0, &corrfunc)
        .unwrap();

    let fim = result[0];

    let fim_clean = fim;

    println!("{:.5}", fim_clean);
    println!("{:.5}", fim_clean.pseudo_inverse(0.0001).unwrap());
    println!(
        "{:?}",
        <std::vec::Vec<f64> as std::convert::TryInto<[f64; 12]>>::try_into(
            fim_clean
                .pseudo_inverse(0.00001)
                .unwrap()
                .diagonal()
                .iter()
                .map(|v| v.sqrt())
                .collect::<Vec<f64>>()
        )
        .unwrap()
    );

    // Create synthetic output.
    let mut output = DMatrix::<ObserVec<_, 3>>::zeros(sc.len(), 1);

    model
        .fevm_initialize_states_only(&sc, &mut fevmd_0)
        .unwrap();
    model
        .fevm_simulate(&sc, &mut fevmd_0, &mut output, None)
        .unwrap();

    let sc_synth = ScObsSeries::<_, ObserVec<_, 3>>::from_iterator((0..20).map(|i| {
        ScObs::new(
            i as f64 * 1.0 * 3600.0,
            ScObsConf::Distance(1.0),
            if output.row(i)[0].any_nan() {
                None
            } else {
                Some(output.row(i)[0].clone())
            },
        )
    }));

    const ENSEMBLE_SIZE: usize = 4096;

    let mut pfsettings = ParticleFilterSettingsBuilder::<_, 3>::default()
        .exploration_factor(1.5)
        .noise(FEVMNoise::Gaussian(NOISE_VAR.sqrt(), 1))
        .quantiles([0.2, 0.5, 1.0])
        .rseed(70)
        .time_limit(1.5)
        .build()
        .unwrap();

    let init_result = model
        .pf_initialize_ensemble(
            &sc_synth,
            ENSEMBLE_SIZE,
            2_usize.pow(18),
            Some((&root_mean_square_filter, 9.5)),
            &mut pfsettings,
        )
        .unwrap();

    init_result
        .write(format!(
            "/Users/ajweiss/Documents/Data/ocnus_fisher/{:03}.particles",
            0
        ))
        .unwrap();

    let mut total_counter = 1;
    let mut fevmd = init_result.fevmd;
    let mut epsilon = init_result.error_quantiles.unwrap()[0];

    for _ in 1..20 {
        let run = model.abcpf_run(
            &sc_synth,
            &fevmd,
            ENSEMBLE_SIZE,
            2_usize.pow(16),
            ABCParticleFilterMode::Threshold((&root_mean_square_filter, epsilon)),
            &mut pfsettings,
        );

        let run_pfresult = match run {
            Ok(result) => result,
            Err(err) => match err {
                FEVMError::ParticleFilter(pferr) => match pferr {
                    ParticleFilterError::TimeLimitExceeded { .. } => break,
                    _ => {
                        pfsettings.rseed += 1;
                        continue;
                    }
                },
                _ => panic!(),
            },
        };

        run_pfresult
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_fisher/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = run_pfresult.fevmd;
        total_counter += 1;
        epsilon = run_pfresult.error_quantiles.unwrap()[0];
    }

    info!("switching to bootstrap");

    for siri in 0..10 {
        let mut bssettings = ParticleFilterSettingsBuilder::<_, 3>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoise::Multivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(
                        sc_synth.len(),
                        sc_synth.len(),
                        NOISE_VAR * ((5 * (10 - siri) + 1) as f64),
                    )
                    .as_view(),
                )
                .unwrap(),
                0,
            ))
            .build()
            .unwrap();

        let result = model.sirpf_run(&sc_synth, &fevmd, ENSEMBLE_SIZE, 32 * 4096, &mut bssettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_fisher/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = pfres.fevmd;
        total_counter += 1;
    }
}
