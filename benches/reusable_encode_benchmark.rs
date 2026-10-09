//! Per-block encode + repair at the production geometry (K=128, T=1344,
//! R=26): `ReusableSourceBlockEncoder` against `SourceBlockEncoder`.
//!
//! `cargo bench --bench reusable_encode_benchmark`

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use raptorq::{
    ObjectTransmissionInformation, ReusableSourceBlockEncoder, SourceBlockEncoder,
    SourceBlockEncodingPlan,
};

const K: u32 = 128;
const T: u16 = 1344;
const R: u32 = 26;
/// The short final block of an object, closed early.
const SHORT: u32 = 51;
/// A per-symbol record header, as the fec-raptorq FFI writes.
const HEADER: usize = 8;

fn data(len: usize) -> Vec<u8> {
    let mut x: u32 = 0x9E37_79B9;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let config = ObjectTransmissionInformation::new(0, T, 1, 1, 8);
    let t = usize::from(T);
    let full = data(K as usize * t);
    let short = data(SHORT as usize * t - 100);
    let plan = Arc::new(SourceBlockEncodingPlan::generate(K as u16));
    let stride = HEADER + t;
    let mut out = vec![0u8; R as usize * stride];

    let mut group = c.benchmark_group("encode_repair_k128_t1344_r26");
    group.throughput(Throughput::Bytes(full.len() as u64));

    group.bench_function("source_block_encoder_planned", |b| {
        b.iter(|| {
            let encoder = SourceBlockEncoder::with_encoding_plan(0, &config, &full, &plan);
            black_box(encoder.repair_packets(0, R));
        })
    });

    let mut encoder = ReusableSourceBlockEncoder::with_plan(&config, plan.clone()).unwrap();
    group.bench_function("reusable_planned", |b| {
        b.iter(|| {
            encoder.encode_block(black_box(&full), K).unwrap();
            encoder
                .repair_range_into(K, R, &mut out[HEADER..], stride)
                .unwrap();
            black_box(&out);
        })
    });

    let mut planless = ReusableSourceBlockEncoder::new(&config).unwrap();
    group.bench_function("reusable_planless", |b| {
        b.iter(|| {
            planless.encode_block(black_box(&full), K).unwrap();
            planless
                .repair_range_into(K, R, &mut out[HEADER..], stride)
                .unwrap();
            black_box(&out);
        })
    });

    group.bench_function("reusable_repair_only", |b| {
        b.iter(|| {
            encoder
                .repair_range_into(K, R, &mut out[HEADER..], stride)
                .unwrap();
            black_box(&out);
        })
    });
    group.finish();

    let mut group = c.benchmark_group("encode_repair_short_k51_t1344_r26");
    group.throughput(Throughput::Bytes(short.len() as u64));
    let mut padded = short.clone();
    padded.resize(SHORT as usize * t, 0);
    group.bench_function("source_block_encoder_new", |b| {
        b.iter(|| {
            let encoder = SourceBlockEncoder::new(0, &config, &padded);
            black_box(encoder.repair_packets(0, R));
        })
    });
    group.bench_function("reusable_full_block_plan_attached", |b| {
        b.iter(|| {
            encoder.encode_block(black_box(&short), SHORT).unwrap();
            encoder
                .repair_range_into(SHORT, R, &mut out[HEADER..], stride)
                .unwrap();
            black_box(&out);
        })
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
