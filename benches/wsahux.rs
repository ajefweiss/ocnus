use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::DMatrix;
use ocnus::{
    base::{OcnusEnsbl, OcnusModel, ScObs, ScObsConf, ScObsSeries},
    models::WSAHUXModel,
    obser::{MeasureInSituPlasmaBulkVelocity, NullNoise, ObserVec},
    stats::{ConstantDensity, Density, MultivariateDensity},
};
use std::{hint::black_box, path::Path, time::Duration};

const ENSEMBLE_SIZE: usize = 2_usize.pow(12);

fn benchmark_wsahux_f32(c: &mut Criterion) {
    let prior = MultivariateDensity::<_, 8>::new(&[
        ConstantDensity::new(285.0),     // a1 = 285 # speeed
        ConstantDensity::new(625.0),     // a2 = 625 # speed
        ConstantDensity::new(2.0 / 9.0), // a3 = 2/9 # expansion factor coefficient
        ConstantDensity::new(1.0),       // a4 = 1 # exp offset
        ConstantDensity::new(0.8),       // a5 = 0.8 # exp factor
        ConstantDensity::new(2.0),       // a6 = 2 # distance fraction (DO NOT CHANGE)
        ConstantDensity::new(2.0),       // a7 = 2 # distance factor coefficient
        ConstantDensity::new(3.0),       // a8 = 3 # coefficient everything
    ]);

    let path = Path::new("examples")
        .join("data")
        .join("wsapy_NSO-GONG_CR2047_0_NSteps90.json");

    let range = (&prior).get_range();
    let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

    model.limit_latitude(1.0);

    let sc = ScObsSeries::<f32>::from_iterator(
        (0..150).map(|i| ScObs::new((14400 * i) as f32, ScObsConf::Position([1.0, 0.0, 0.0]))),
    );

    let mut group = c.benchmark_group("wsahux_bench");

    let mut ensbl = OcnusEnsbl::new(ENSEMBLE_SIZE, range);
    let mut output = DMatrix::<ObserVec<_, 1>>::zeros(sc.len(), ENSEMBLE_SIZE);

    group
        .significance_level(0.05)
        .sample_size(50)
        .measurement_time(Duration::from_secs(5));

    group.throughput(Throughput::Elements(ENSEMBLE_SIZE as u64));
    group.bench_function("wsahux_initialize", |b| {
        b.iter(|| {
            model
                .initialize_ensbl::<100, _>(
                    black_box(&mut ensbl),
                    black_box(None::<&MultivariateDensity<f32, 8>>),
                    42,
                )
                .unwrap();
        });
    });

    group.throughput(Throughput::Elements((ENSEMBLE_SIZE * sc.len()) as u64));
    group.bench_function("wsahux_simulate", |b| {
        b.iter(|| {
            model
                .initialize_ensbl::<100, _>(
                    black_box(&mut ensbl),
                    black_box(None::<&MultivariateDensity<f32, 8>>),
                    42,
                )
                .unwrap();
            model
                .simulate_ensbl(
                    &sc,
                    &mut ensbl,
                    &WSAHUXModel::<f32, 215, MultivariateDensity<f32, 8>>::observe_pbv,
                    &mut output.as_view_mut(),
                    None::<&mut NullNoise<f32>>,
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_wsahux_f32);
criterion_main!(benches);
