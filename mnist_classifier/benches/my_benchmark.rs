use criterion::{criterion_group, criterion_main, Criterion};
use mnist_classifier::{evaluate, evaluate_threaded};


fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("eval mnist", |b| b.iter(|| evaluate(false)));
    c.bench_function("eval mnist threaded", |b| b.iter(|| evaluate_threaded(false)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);