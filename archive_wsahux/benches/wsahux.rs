#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Dyn, OMatrix, SVector, U8, Vector3};
use ocnus::{
    base::{Model, ModelEnsbl},
    instr::Plasma,
    obs::{Obs, ObsEnsbl, conf::VecConf, noise::NullNoise},
};
use ocnus_swm::models::WSAHUXModel;
use prodef::multivariate::{ConstantDensity, MultivariateDensity};
use std::{hint::black_box, path::Path, time::Duration};

const ENSEMBLE_SIZE: usize = 2_usize.pow(12);

fn benchmark_wsahux_f32(c: &mut Criterion) {
    let prior = MultivariateDensity::new(SVector::from([
        ConstantDensity::new(285.0).into(),     // a1 = 285 # speeed
        ConstantDensity::new(625.0).into(),     // a2 = 625 # speed
        ConstantDensity::new(2.0 / 9.0).into(), // a3 = 2/9 # expansion factor coefficient
        ConstantDensity::new(1.0).into(),       // a4 = 1 # exp offset
        ConstantDensity::new(0.8).into(),       // a5 = 0.8 # exp factor
        ConstantDensity::new(2.0).into(),       // a6 = 2 # distance fraction (DO NOT CHANGE)
        ConstantDensity::new(2.0).into(),       // a7 = 2 # distance factor coefficient
        ConstantDensity::new(3.0).into(),       // a8 = 3 # coefficient everything
    ]));

    let path = Path::new("examples/data").join("nso_gong_CR2047.json");

    let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior.clone(), path, 2.0).unwrap();

    model.limit_latitude(1.0);

    let obs = Obs::from_iter(
        (0..150).map(|i| VecConf::new((14400 * i) as f32, Vector3::new(1.0, 0.0, 0.0))),
    );

    let mut model_ensbl =
        ModelEnsbl::new(OMatrix::<f32, U8, Dyn>::zeros(ENSEMBLE_SIZE), None, None);
    let mut obs_ensbl = ObsEnsbl::new(obs.clone(), ENSEMBLE_SIZE, None).unwrap();

    let mut group = c.benchmark_group("core_bench");

    group
        .significance_level(0.05)
        .sample_size(100)
        .measurement_time(Duration::from_secs(25));

    group.throughput(Throughput::Elements(ENSEMBLE_SIZE as u64));
    group.bench_function("wsahux_initialize", |b| {
        b.iter(|| {
            model
                .initialize_ensbl(
                    black_box(&mut model_ensbl),
                    black_box(prior.clone()),
                    1000,
                    42,
                )
                .unwrap();
        });
    });

    group.throughput(Throughput::Elements((ENSEMBLE_SIZE * obs.len()) as u64));
    group.bench_function("wsahux_simulate", |b| {
        b.iter(|| {
            model
                .initialize_ensbl(
                    black_box(&mut model_ensbl),
                    black_box(prior.clone()),
                    1000,
                    42,
                )
                .unwrap();
            model
                .simulate_ensbl(
                    &mut model_ensbl,
                    &mut obs_ensbl,
                    &WSAHUXModel::<f32, 215, MultivariateDensity<f32, U8>>::observe_pbs,
                    &mut None::<&mut NullNoise<f32>>,
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_wsahux_f32);
criterion_main!(benches);
