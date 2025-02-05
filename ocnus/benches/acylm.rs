use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use ndarray::Array2;
use ocnus::{
    fevms::{CCLFF_Model, FEVMEnsbl, ForwardEnsembleVectorModel, ModelObserVec},
    stats::{ConstantPDF, PUnivariatePDF, UniformPDF, UnivariatePDF},
    Fp, ScConf, ScObs,
};
use std::time::Duration;

const ENSEMBLE_SIZE: usize = 2_usize.pow(18);

fn benchmark_cc_lff(c: &mut Criterion) {
    let model = CCLFF_Model(PUnivariatePDF::new([
        UnivariatePDF::Uniform(UniformPDF::new((-90.0, 90.0)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.0, 360.0)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
        UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
        UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
        UnivariatePDF::Constant(ConstantPDF::new(2.4)),
        UnivariatePDF::Uniform(UniformPDF::new((0.7, 1.0)).unwrap()),
    ]));

    #[allow(clippy::excessive_precision)]
    let refobs = [
        ModelObserVec::from([-2.67159853, -13.38264243, -7.20006469]),
        ModelObserVec::from([-3.13531505, -13.98423491, -6.64658269]),
        ModelObserVec::from([-5.34451816, -14.85328723, -1.73480069]),
        ModelObserVec::from([-3.38360946, -15.34720671, -1.71139834]),
        ModelObserVec::from([-4.00239361, -14.94369, 1.2604304]),
        ModelObserVec::from([-4.21540068, -13.93568105, 4.73878965]),
        ModelObserVec::from([-2.7273795, -14.29364075, 4.8267508]),
        ModelObserVec::from([-5.41469694, -13.9772912, 4.14112123]),
        ModelObserVec::from([-4.28371012, -13.89409455, 5.13879915]),
        ModelObserVec::from([-4.30711573, -12.61217154, 5.78382821]),
    ];

    let sc = ScObs::<ModelObserVec<3>>::from_iterator((0..refobs.len()).map(|i| {
        (
            Some(refobs[i].clone()),
            ScConf::TimeDistance(10800.0 + i as Fp * 2.0 * 3600.0, 1.0),
        )
    }));

    let mut data = FEVMEnsbl::new(ENSEMBLE_SIZE);
    let mut output = Array2::<Option<ModelObserVec<3>>>::default((sc.len(), ENSEMBLE_SIZE));

    let mut group = c.benchmark_group("cc_lff");

    group
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(5))
        .sample_size(100);

    group.throughput(Throughput::Elements((sc.len() * ENSEMBLE_SIZE) as u64));

    group.bench_function("cc_lff_all", |b| {
        b.iter(|| {
            model
                .fevm_initialize(
                    sc.as_scconf_slice(),
                    &mut data,
                    None::<&PUnivariatePDF<8>>,
                    42,
                )
                .unwrap();

            model
                .fevm_simulate(sc.as_scconf_slice(), &mut data, &mut output.view_mut())
                .unwrap();
        });
    });

    group.throughput(Throughput::Elements(ENSEMBLE_SIZE as u64));

    group.bench_function("cc_lff_initialize", |b| {
        b.iter(|| {
            model
                .fevm_initialize(
                    sc.as_scconf_slice(),
                    &mut data,
                    None::<&PUnivariatePDF<8>>,
                    42,
                )
                .unwrap();
        });
    });

    group.bench_function("cc_lff_initialize_params_only", |b| {
        b.iter(|| {
            model
                .fevm_initialize_params_only(&mut data, None::<&PUnivariatePDF<8>>, 42)
                .unwrap();
        });
    });

    group.bench_function("cc_lff_initialize_states_only", |b| {
        b.iter(|| {
            model
                .fevm_initialize_states_only(sc.as_scconf_slice(), &mut data)
                .unwrap();
        });
    });

    group.throughput(Throughput::Elements((sc.len() * ENSEMBLE_SIZE) as u64));

    model
        .fevm_initialize(
            sc.as_scconf_slice(),
            &mut data,
            None::<&PUnivariatePDF<8>>,
            42,
        )
        .unwrap();

    group.bench_function("cc_lff_simulate", |b| {
        b.iter(|| {
            model
                .fevm_simulate(sc.as_scconf_slice(), &mut data, &mut output.view_mut())
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_cc_lff);
criterion_main!(benches);
