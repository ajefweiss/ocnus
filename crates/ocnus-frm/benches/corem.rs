#![allow(missing_docs)]

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Dyn, OMatrix, SVector, U11, Vector3};
use ocnus::{
    base::{Model, ModelEnsbl},
    instr::Magnetometer,
    obs::{Obs, ObsEnsbl, conf::VecConf, data::ObsVec, noise::NullNoise},
};
use ocnus_frm::models::COREModel;
use prodef::multivariate::{ConstantDensity, MultivariateDensity, UniformDensity};
use std::{hint::black_box, time::Duration};

const ENSEMBLE_SIZE: usize = 2_usize.pow(16);

fn benchmark_core_f32(c: &mut Criterion) {
    let prior = MultivariateDensity::new(SVector::from([
        UniformDensity::new(-1.0, 1.0).unwrap().into(),
        UniformDensity::new(0.5, 1.0).unwrap().into(),
        UniformDensity::new(-0.5, 0.5).unwrap().into(),
        ConstantDensity::new(20.0).into(),
        UniformDensity::new(0.05, 0.25).unwrap().into(),
        ConstantDensity::new(1.0).into(),
        ConstantDensity::new(1125.0).into(),
        UniformDensity::new(5.0, 100.0).unwrap().into(),
        UniformDensity::new(-10.0, 10.0).unwrap().into(),
        ConstantDensity::new(400.0).into(),
        ConstantDensity::new(1.0).into(),
    ]));

    let model = COREModel::new(prior.clone());

    #[allow(clippy::excessive_precision)]
    let ref_data = [
        ObsVec::from([-2.67159853, -13.38264243, -7.20006469]),
        ObsVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ObsVec::from([-5.34451816, -14.85328723, -1.73480069]),
        ObsVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ObsVec::from([-4.00239361, -14.94369, 1.2604304]),
        ObsVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ObsVec::from([-2.7273795, -14.29364075, 4.8267508]),
        ObsVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ObsVec::from([-4.28371012, -13.89409455, 5.13879915]),
        ObsVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    let obs = Obs::from_iter((0..ref_data.len()).map(|i| {
        VecConf::new(
            224640.0 + i as f32 * 3600.0 * 2.0,
            Vector3::new(1.0, 0.0, 0.0),
        )
    }));

    let mut model_ensbl =
        ModelEnsbl::new(OMatrix::<f32, U11, Dyn>::zeros(ENSEMBLE_SIZE), None, None);
    let mut obs_ensbl = ObsEnsbl::new(obs.clone(), ENSEMBLE_SIZE, None).unwrap();

    let mut group = c.benchmark_group("core_bench");

    group
        .significance_level(0.05)
        .sample_size(100)
        .measurement_time(Duration::from_secs(25));

    group.throughput(Throughput::Elements(ENSEMBLE_SIZE as u64));
    group.bench_function("core_initialize", |b| {
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
    group.bench_function("core_simulate", |b| {
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
                .simulate_mag3(
                    &mut model_ensbl,
                    &mut obs_ensbl,
                    &mut None::<&mut NullNoise<f32>>,
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_core_f32);
criterion_main!(benches);
