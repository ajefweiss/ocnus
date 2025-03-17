use chrono::Local;
use core::f32;
use env_logger::Builder;
use log::info;
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
use ocnus::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{
        CCLFFModel, FEVM, FEVMData, FEVMError, FEVMNullState, FisherInformation,
        filters::{
            ABCParticleFilter, ABCParticleFilterMode, BSParticleFilter, ParticleFilter,
            ParticleFilterError, ParticleFilterSettingsBuilder, root_mean_square_filter,
        },
        noise::{FEVMNoiseGaussian, FEVMNoiseMultivariate, FEVMNoiseNull},
    },
    geometry::XCState,
    obser::ObserVec,
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

    let model = CCLFFModel(PDFUnivariates::new([
        PDFUniform::new_uvpdf((-(90.0_f32.to_radians()), (90.0_f32.to_radians()))).unwrap(),
        PDFUniform::new_uvpdf((0.0, f32::consts::TAU)).unwrap(),
        PDFUniform::new_uvpdf((-0.75, 0.75)).unwrap(),
        PDFReciprocal::new_uvpdf((0.05, 0.35)).unwrap(),
        PDFConstant::new_uvpdf(1125.0),
        PDFUniform::new_uvpdf((5.0, 25.0)).unwrap(),
        PDFUniform::new_uvpdf((-2.4, 2.4)).unwrap(),
        PDFUniform::new_uvpdf((0.7, 1.0)).unwrap(),
    ]));

    let sc = ScObsSeries::<ObserVec<3>>::from_iterator(
        (0..20).map(|i| ScObs::new(i as f32 * 1.0 * 3600.0, ScObsConf::Distance(1.0), None)),
    );

    let mut fevmd_0 = FEVMData {
        params: Matrix::<f32, Const<8>, Dyn, VecStorage<f32, Const<8>, Dyn>>::from_iterator(
            1,
            [
                -15.3943205_f32.to_radians(),
                177.20819_f32.to_radians(),
                -0.03788574,
                0.1880269,
                1125.0,
                15.92187,
                1.8,
                0.8245641,
            ],
        ),
        fevm_states: vec![FEVMNullState::default(); 1],
        geom_states: vec![XCState::default(); 1],
        weights: vec![1.0],
    };

    const NOISE_VAR: f32 = 1.0;

    let corrfunc = |d| if d == 0.0 { NOISE_VAR } else { 0.0 };

    let result = model
        .fischer_information_matrix(&sc, &fevmd_0, &corrfunc)
        .unwrap();

    let fim = result[0];

    let fim_clean = fim
        .remove_column(7)
        .remove_column(6)
        .remove_column(4)
        .remove_row(7)
        .remove_row(6)
        .remove_row(4);

    println!("{:.5}", fim_clean);
    println!("{:.5}", fim_clean.try_inverse().unwrap());
    println!(
        "{:?}",
        fim_clean
            .try_inverse()
            .unwrap()
            .diagonal()
            .iter()
            .map(|v| v.sqrt())
            .collect::<Vec<f32>>()
    );

    // Create synthetic output.
    let mut output = DMatrix::<ObserVec<3>>::zeros(sc.len(), 1);

    model
        .fevm_initialize_states_only(&sc, &mut fevmd_0)
        .unwrap();
    model
        .fevm_simulate(
            &sc,
            &mut fevmd_0,
            &mut output,
            None::<(&FEVMNoiseNull, u64)>,
        )
        .unwrap();

    let sc_synth = ScObsSeries::<ObserVec<3>>::from_iterator((0..20).map(|i| {
        ScObs::new(
            i as f32 * 1.0 * 3600.0,
            ScObsConf::Distance(1.0),
            if output.row(i)[0].any_nan() {
                None
            } else {
                Some(output.row(i)[0].clone())
            },
        )
    }));

    const ENSEMBLE_SIZE: usize = 4096;

    let mut pfsettings = ParticleFilterSettingsBuilder::<3, _>::default()
        .exploration_factor(1.5)
        .noise(FEVMNoiseGaussian(NOISE_VAR.sqrt()))
        .quantiles([0.2, 0.5, 1.0])
        .rseed(70)
        .time_limit(1.5)
        .build()
        .unwrap();

    let init_result = model
        .pf_initialize_data(
            &sc_synth,
            ENSEMBLE_SIZE,
            2_usize.pow(18),
            Some((&root_mean_square_filter, 9.5)),
            &mut pfsettings,
        )
        .unwrap();

    init_result
        .write(format!(
            "/Users/ajweiss/Documents/Data/ocnus_fisher_lim/{:03}.particles",
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
                "/Users/ajweiss/Documents/Data/ocnus_fisher_lim/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = run_pfresult.fevmd;
        total_counter += 1;
        epsilon = run_pfresult.error_quantiles.unwrap()[0];
    }

    info!("switching to bootstrap");

    for _ in 0..10 {
        let mut bssettings = ParticleFilterSettingsBuilder::<3, _>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoiseMultivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(sc_synth.len(), sc_synth.len(), NOISE_VAR)
                        .as_view(),
                )
                .unwrap(),
            ))
            .build()
            .unwrap();

        let result = model.bootpf_run(&sc_synth, &fevmd, ENSEMBLE_SIZE, 8 * 4096, &mut bssettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_fisher_lim/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = pfres.fevmd;
        total_counter += 1;
    }
}
