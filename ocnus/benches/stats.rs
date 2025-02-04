use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use nalgebra::Const;
use ocnus::{
    stats::{ConstantPDF, PUnivariatePDF, ProbabilityDensityFunction, UniformPDF, UnivariatePDF},
    PMatrix,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

const ENSEMBLE_SIZE: usize = 2_usize.pow(18);

fn benchmark_sample(c: &mut Criterion) {
    let prior_density = PUnivariatePDF::new([
        UnivariatePDF::Uniform(UniformPDF::new((-90.0, 90.0)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.0, 360.0)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((-0.75, 0.75)).unwrap()),
        UnivariatePDF::Uniform(UniformPDF::new((0.1, 0.25)).unwrap()),
        UnivariatePDF::Constant(ConstantPDF::new(1125.0)),
        UnivariatePDF::Uniform(UniformPDF::new((5.0, 25.0)).unwrap()),
        UnivariatePDF::Constant(ConstantPDF::new(2.4)),
        UnivariatePDF::Uniform(UniformPDF::new((0.7, 1.0)).unwrap()),
    ]);

    let density = ProbabilityDensityFunction::new(prior_density);
    let mut params = PMatrix::<Const<8>>::zeros(ENSEMBLE_SIZE);

    let mut group = c.benchmark_group("sample");

    group.throughput(Throughput::Elements((ENSEMBLE_SIZE) as u64));

    group.bench_function("sample_uvpdf", |b| {
        b.iter(|| {
            params
                .par_column_iter_mut()
                .chunks(128)
                .enumerate()
                .for_each(|(cdx, mut chunks)| {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64((cdx + 42) as u64);

                    chunks.iter_mut().for_each(|col| {
                        col.set_column(0, &density.sample(&mut rng).unwrap());
                    });
                });
        });
    });

    group.finish();
}

criterion_group!(benches, benchmark_sample);
criterion_main!(benches);
