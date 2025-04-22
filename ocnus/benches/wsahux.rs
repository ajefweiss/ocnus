use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use nalgebra::{Const, DMatrix, Dyn, Matrix, VecStorage};
use ocnus::{
    forward::{FSMEnsbl, OcnusFSM, WSAHUXModel, WSAState},
    obser::{NoNoise, ObserVec, ScObs, ScObsConf, ScObsSeries},
    prodef::{Constant1D, UnivariateND},
};
use std::{hint::black_box, path::Path, time::Duration};

const ENSEMBLE_SIZE: usize = 2_usize.pow(8);

fn benchmark_wsahux_f32(c: &mut Criterion) {
    let prior = UnivariateND::new([
        Constant1D::new(285.0),     // a1 = 285 # speeed
        Constant1D::new(625.0),     // a2 = 625 # speed
        Constant1D::new(2.0 / 9.0), // a3 = 2/9 # expansion factor coefficient
        Constant1D::new(1.0),       // a4 = 1 # exp offset
        Constant1D::new(0.8),       // a5 = 0.8 # exp factor``
        Constant1D::new(2.0),       // a6 = 2 # distance fraction (DO NOT CHANGE)
        Constant1D::new(2.0),       // a7 = 2 # distance factor coefficient
        Constant1D::new(3.0),       // a8 = 3 # coefficient everything
    ]);

    let path = Path::new("examples")
        .join("data")
        .join("wsapy_NSO-GONG_CR2047_0_NSteps90.json");

    let mut model = WSAHUXModel::<f32, 215, _>::from_file(prior, path, 2.0).unwrap();

    model.limit_latitude(1.0);

    let sc = ScObsSeries::<f32, ObserVec<f32, 1>>::from_iterator(
        (0..150).map(|i| ScObs::new((14400 * i) as f32, ScObsConf::Distance(1.0), None)),
    );

    let mut group = c.benchmark_group("wsahux_bench");

    // Create temporary simulation data and out arrays, if necessary.
    let mut data = FSMEnsbl {
        params_array: Matrix::<f32, Const<8>, Dyn, VecStorage<f32, Const<8>, Dyn>>::zeros(
            ENSEMBLE_SIZE,
        ),
        fm_states: vec![WSAState::default(); ENSEMBLE_SIZE],
        cs_states: vec![(); ENSEMBLE_SIZE],
        weights: vec![1.0 / ENSEMBLE_SIZE as f32; ENSEMBLE_SIZE],
    };

    let mut output = DMatrix::<ObserVec<f32, 1>>::zeros(sc.len(), ENSEMBLE_SIZE);

    group
        .significance_level(0.05)
        .sample_size(50)
        .measurement_time(Duration::from_secs(5));

    group.throughput(Throughput::Elements(ENSEMBLE_SIZE as u64));
    group.bench_function("wsahux_initialize", |b| {
        b.iter(|| {
            model
                .fsm_initialize_ensbl(
                    black_box(&sc),
                    black_box(&mut data),
                    black_box(None::<&UnivariateND<f32, 8>>),
                    42,
                )
                .unwrap();
        });
    });

    group.throughput(Throughput::Elements((ENSEMBLE_SIZE * sc.len()) as u64));
    group.bench_function("wsahux_simulate", |b| {
        b.iter(|| {
            model
                .fsm_initialize_ensbl(
                    black_box(&sc),
                    black_box(&mut data),
                    black_box(None::<&UnivariateND<f32, 8>>),
                    42,
                )
                .unwrap();
            model
                .fsm_simulate_ensbl(
                    &sc,
                    &mut data,
                    &mut output.as_view_mut(),
                    None::<&mut NoNoise<f32>>,
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_wsahux_f32);
criterion_main!(benches);
