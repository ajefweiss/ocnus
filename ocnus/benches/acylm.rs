use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
use ocnus::{
    ScObs, ScObsConf, ScObsSeries,
    fevm::{CCLFFModel, FEVM, FEVMData, FEVMNoiseZero, FEVMNullState},
    geometry::XCState,
    obser::ObserVec,
    stats::{PDFConstant, PDFUniform, PDFUnivariates},
};
use std::{hint::black_box, time::Duration};

const ENSEMBLE_SIZE: usize = 2_usize.pow(18);

fn benchmark_lff(c: &mut Criterion) {
    let prior = PDFUnivariates::new([
        PDFUniform::new_uvpdf((-1.0, 1.0)).unwrap(),
        PDFUniform::new_uvpdf((0.5, 1.0)).unwrap(),
        PDFUniform::new_uvpdf((0.05, 0.1)).unwrap(),
        PDFUniform::new_uvpdf((0.1, 0.5)).unwrap(),
        PDFConstant::new_uvpdf(1125.0),
        PDFUniform::new_uvpdf((5.0, 100.0)).unwrap(),
        PDFUniform::new_uvpdf((-2.4, 2.4)).unwrap(),
        PDFUniform::new_uvpdf((0.0, 1.0)).unwrap(),
    ]);

    let model = CCLFFModel(prior);

    #[allow(clippy::excessive_precision)]
    let refobs = [
        ObserVec::from([-2.67159853, -13.38264243, -7.20006469]),
        ObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObserVec::from([-5.34451816, -14.85328723, -1.73480069]),
        ObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObserVec::from([-4.00239361, -14.94369, 1.2604304]),
        ObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObserVec::from([-2.7273795, -14.29364075, 4.8267508]),
        ObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObserVec::from([-4.28371012, -13.89409455, 5.13879915]),
        ObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    let sc = ScObsSeries::<ObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        ScObs::new(
            224640.0 + i as f32 * 3600.0 * 2.0,
            ScObsConf::Distance(1.0),
            Some(refobs[i].clone()),
        )
    }));

    let mut group = c.benchmark_group("acylm_lff_bench");

    // Create temporary simulation data and out arrays, if necessary.
    let mut data = FEVMData {
        params: Matrix::<f32, Const<8>, Dyn, VecStorage<f32, Const<8>, Dyn>>::zeros(ENSEMBLE_SIZE),
        fevm_states: vec![FEVMNullState::default(); ENSEMBLE_SIZE],
        geom_states: vec![XCState::default(); ENSEMBLE_SIZE],
        rseed: 42,
    };

    let mut output = DMatrix::<ObserVec<3>>::zeros(sc.len(), ENSEMBLE_SIZE);

    group
        .significance_level(0.05)
        .sample_size(100)
        .measurement_time(Duration::from_secs(10));

    group.throughput(Throughput::Elements((ENSEMBLE_SIZE * sc.len()) as u64));
    group.bench_function("acylm_lff", |b| {
        b.iter(|| {
            model
                .fevm_initialize(
                    black_box(&sc),
                    black_box(&mut data),
                    black_box(None::<&&PDFUnivariates<8>>),
                )
                .unwrap();
            model
                .fevm_simulate(&sc, &mut data, &mut output, None::<&FEVMNoiseZero>)
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_lff);
criterion_main!(benches);
