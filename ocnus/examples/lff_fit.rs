use chrono::Local;
use env_logger::Builder;
use nalgebra::DMatrix;
use ocnus::{
    fevms::{
        ABCMode, ABCParticleFilter, ABCSettingsBuilder, ABCSummaryStatistic, ModelObserVec,
        OCModel_CC_LFF, ParticleFiltering,
    },
    stats::{ConstantPDF, CovMatrix, PUnivariatePDF, UniformPDF, UnivariatePDF},
    Fp, ScConf, ScObs,
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

    let model: OCModel_CC_LFF<_> = OCModel_CC_LFF(PUnivariatePDF::new([
        UnivariatePDF::Uniform(
            UniformPDF::new((-std::f64::consts::PI as Fp, std::f64::consts::PI as Fp)).unwrap(),
        ),
        UnivariatePDF::Uniform(UniformPDF::new((0.0, 2.0 * std::f64::consts::PI as Fp)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
        UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
        UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((-2.4, 2.4)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.7, 1.0)).unwrap()),
    ]));

    #[allow(clippy::excessive_precision)]
    let refobs = [
        // None,
        Some(ModelObserVec::from([
            -2.67159853,
            -13.38264243,
            -7.20006469,
        ])),
        // Some(ModelObserVec::from([-3.13531505, -13.98423491, -6.64658269])),
        Some(ModelObserVec::from([
            -5.34451816,
            -14.85328723,
            -1.73480069,
        ])),
        // ModelObserVec::from([-3.38360946, -15.34720671, -1.71139834])),
        Some(ModelObserVec::from([-4.00239361, -14.94369, 1.2604304])),
        // ModelObserVec::from([-4.21540068, -13.93568105, 4.73878965])),
        Some(ModelObserVec::from([-2.7273795, -14.29364075, 4.8267508])),
        // ModelObserVec::from([-5.41469694, -13.9772912, 4.14112123])),
        Some(ModelObserVec::from([-4.28371012, -13.89409455, 5.13879915])),
        // ModelObserVec::from([-4.30711573, -12.61217154, 5.78382821])),
        // None,
    ];

    let sc = ScObs::<ModelObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        (
            refobs[i].clone(),
            ScConf::TimeDistance(10800.0 + i as Fp * 2.0 * 3600.0, 1.0),
        )
    }));

    let covm = CovMatrix::from_matrix(DMatrix::<Fp>::from_diagonal_element(
        refobs.len(),
        refobs.len(),
        1.0,
    ))
    .unwrap();

    let mut filter = model.fevmfilter_builder().build().unwrap();

    let mut settings = ABCSettingsBuilder::default()
        .initial_seed(43)
        .target_size(2 * 1024)
        .simulation_size(64 * 1024)
        .exploration_factor(1.5)
        .noise_covm(Some(covm))
        .mode(ABCMode::Threshold(0.25))
        .summary_statistic(ABCSummaryStatistic::MeanSquareError)
        .build()
        .unwrap();

    let _ = model.abc_filter(&sc, &mut filter, &settings).unwrap();

    settings.mode = ABCMode::Threshold(0.15);

    let _ = model.abc_filter(&sc, &mut filter, &settings).unwrap();
    // println!("{:?}", res.result);
    // for i in 0..5 {
    //     settings.mode = ABCMode::Threshold(1.0 / (i as Fp + 2.0));

    //     let _ = model.abc_filter(&sc, &mut filter, &settings).unwrap();
    // }
}
