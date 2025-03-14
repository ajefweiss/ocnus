use chrono::Local;
use core::f64;
use env_logger::Builder;
use nalgebra::DMatrix;
use ocnus::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{
        CCLFFModel,
        filters::{
            ABCParticleFilter, ABCParticleFilterMode, MVLHParticleFilter, ParticleFilter,
            ParticleFilterResults, ParticleFilterSettingsBuilder, mean_square_normalized_filter,
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
                Local::now().format("%Y-%m-%dT%H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info)
        .init();

    let model = CCLFFModel(PDFUnivariates::new([
        PDFUniform::new_uvpdf((-2.0, 1.0)).unwrap(),
        PDFUniform::new_uvpdf((2.0, 4.2)).unwrap(),
        PDFUniform::new_uvpdf((-0.75, 0.75)).unwrap(),
        PDFReciprocal::new_uvpdf((0.05, 0.35)).unwrap(),
        PDFConstant::new_uvpdf(750.0),
        PDFUniform::new_uvpdf((5.0, 25.0)).unwrap(),
        PDFUniform::new_uvpdf((0.0, 2.4)).unwrap(),
        PDFUniform::new_uvpdf((0.5, 1.0)).unwrap(),
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

    let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            10800.0 + i as f64 * 3600.0 * 2.0,
            ScObsConf::Distance(1.0),
            Some(refobs[i].clone()),
        )
    }));

    let mut fevmd = model
        .pf_initialize_data(
            &sc,
            8192,
            2_usize.pow(18),
            Some((&mean_square_normalized_filter, 0.9)),
            43,
        )
        .unwrap();

    let mut file = std::fs::File::create(format!(
        "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
        0
    ))
    .expect("could not create file!");

    let list_as_json = serde_json::to_string(&ParticleFilterResults {
        fevmd: fevmd.clone(),
        output: DMatrix::<ObserVec<3>>::zeros(1, 1),
        error_quantiles: [0.0; 3],
        errors: Vec::new(),
    })
    .unwrap();

    file.write_all(list_as_json.as_bytes())
        .expect("Cannot write to the file!");

    let mut pfsettings = ParticleFilterSettingsBuilder::<3, _>::default()
        .exploration_factor(1.5)
        .noise(FEVMNoiseGaussian(2.0))
        .build()
        .unwrap();

    for idx in 1..5 {
        let result = model.abcpf_run(
            &sc,
            &fevmd,
            8192,
            2_usize.pow(18),
            ABCParticleFilterMode::AcceptanceRate((&mean_square_normalized_filter, 0.0025)),
            &mut pfsettings,
        );

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                idx
            ))
            .unwrap();

        fevmd = pfres.fevmd;
    }

    for idx in 0..10 {
        let mut mvlhsettings = ParticleFilterSettingsBuilder::<3, _>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoiseMultivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(sc.len(), sc.len(), 10.0 / (idx as f64 + 1.0))
                        .as_view(),
                )
                .unwrap(),
            ))
            .build()
            .unwrap();

        let result = model.mvlhpf_run(&sc, &fevmd, 8192, 2_usize.pow(16), &mut mvlhsettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                5 + idx
            ))
            .unwrap();

        fevmd = pfres.fevmd;
    }

    for idx in 0..10 {
        let mut mvlhsettings = ParticleFilterSettingsBuilder::<3, _>::default()
            .exploration_factor(1.0)
            .noise(FEVMNoiseMultivariate(
                CovMatrix::from_matrix(
                    &DMatrix::from_diagonal_element(sc.len(), sc.len(), 1.0).as_view(),
                )
                .unwrap(),
            ))
            .build()
            .unwrap();

        let result = model.mvlhpf_run(&sc, &fevmd, 8192, 2_usize.pow(16), &mut mvlhsettings);

        let pfres = result.unwrap();

        pfres
            .write(format!(
                "/Users/ajweiss/Documents/Data/ocnus_pf/{:03}.particles",
                15 + idx
            ))
            .unwrap();

        fevmd = pfres.fevmd;
    }
}
