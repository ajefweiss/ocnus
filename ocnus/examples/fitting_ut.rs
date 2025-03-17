use chrono::Local;
use core::f32;
use env_logger::Builder;
use nalgebra::DMatrix;
use ocnus::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{
        CCUTModel, FEVMError,
        filters::{
            ABCParticleFilter, ABCParticleFilterMode, BSParticleFilter, ParticleFilter,
            ParticleFilterError, ParticleFilterSettingsBuilder, root_mean_square_filter,
        },
        noise::{FEVMNoiseGaussian, FEVMNoiseMultivariate},
    },
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

    let model = CCUTModel(PDFUnivariates::new([
        PDFUniform::new_uvpdf((-(90.0_f32.to_radians()), (90.0_f32.to_radians()))).unwrap(),
        PDFUniform::new_uvpdf((0.0, f32::consts::TAU)).unwrap(),
        PDFUniform::new_uvpdf((-0.75, 0.75)).unwrap(),
        PDFReciprocal::new_uvpdf((0.05, 0.35)).unwrap(),
        PDFConstant::new_uvpdf(750.0),
        PDFUniform::new_uvpdf((5.0, 25.0)).unwrap(),
        PDFUniform::new_uvpdf((-5.0, 5.0)).unwrap(),
        PDFUniform::new_uvpdf((0.7, 1.0)).unwrap(),
    ]));

    #[allow(clippy::excessive_precision)]
    let refobs = [
        ObserVec::from([-2.67159853, -13.38264243, -7.20006469]),
        // ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObserVec::from([-5.34451816, -14.85328723, -1.73480069]),
        // ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObserVec::from([-4.00239361, -14.94369, 1.2604304]),
        // ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObserVec::from([-2.7273795, -14.29364075, 4.8267508]),
        // ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObserVec::from([-4.28371012, -13.89409455, 5.13879915]),
        // ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    const ENSEMBLE_SIZE: usize = 4096;

    let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            10800.0 + i as f32 * 3600.0 * 2.0,
            ScObsConf::Distance(1.0),
            Some(refobs[i].clone()),
        )
    }));

    let mut pfsettings = ParticleFilterSettingsBuilder::<3, _>::default()
        .exploration_factor(1.5)
        .noise(FEVMNoiseGaussian(1.0))
        .quantiles([0.2, 0.5, 1.0])
        .rseed(70)
        .time_limit(1.5)
        .build()
        .unwrap();

    let init_result = model
        .pf_initialize_data(
            &sc,
            ENSEMBLE_SIZE,
            2_usize.pow(18),
            Some((&root_mean_square_filter, 9.5)),
            &mut pfsettings,
        )
        .unwrap();

    init_result
        .write(format!(
            "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
            0
        ))
        .unwrap();

    let mut total_counter = 1;
    let mut fevmd = init_result.fevmd;
    let mut epsilon = init_result.error_quantiles.unwrap()[0];

    for _ in 1..20 {
        let run = model.abcpf_run(
            &sc,
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
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = run_pfresult.fevmd;
        total_counter += 1;
        epsilon = run_pfresult.error_quantiles.unwrap()[0];
    }

    for idx in 0..10 {
        let mut bssettings = ParticleFilterSettingsBuilder::<3, _>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoiseMultivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(
                        sc.len(),
                        sc.len(),
                        1.0 + (10.0 / (1 + idx) as f32),
                    )
                    .as_view(),
                )
                .unwrap(),
            ))
            .build()
            .unwrap();

        let result = model.bootpf_run(&sc, &fevmd, ENSEMBLE_SIZE, 2_usize.pow(16), &mut bssettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = pfres.fevmd;
        total_counter += 1;
    }

    for _ in 0..10 {
        let mut bssettings = ParticleFilterSettingsBuilder::<3, _>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoiseMultivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(sc.len(), sc.len(), 1.0).as_view(),
                )
                .unwrap(),
            ))
            .build()
            .unwrap();

        let result = model.bootpf_run(&sc, &fevmd, ENSEMBLE_SIZE, 2_usize.pow(16), &mut bssettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                total_counter
            ))
            .unwrap();

        fevmd = pfres.fevmd;
        total_counter += 1;
    }
}
