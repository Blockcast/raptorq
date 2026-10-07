//! Differential and reuse tests for `ReusableSourceBlockEncoder` and
//! `ReusableSourceBlockDecoder` against `SourceBlockEncoder` and
//! `SourceBlockDecoder`.

use std::sync::Arc;

use raptorq::{
    BlockError, EncodingPacket, ObjectTransmissionInformation, PayloadId,
    ReusableSourceBlockDecoder, ReusableSourceBlockEncoder, SourceBlockDecoder, SourceBlockEncoder,
    SourceBlockEncodingPlan,
};

/// xorshift64*: deterministic, seedable, no dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

/// Debug builds (upstream CI runs `cargo test` unoptimized) get a smaller
/// sample of the slow cases; `cargo test --release` runs all of them.
const FULL: bool = !cfg!(debug_assertions);

fn config(symbol_size: u16, sub_blocks: u16, alignment: u8) -> ObjectTransmissionInformation {
    ObjectTransmissionInformation::new(0, symbol_size, 1, sub_blocks, alignment)
}

/// `SourceBlockEncoder`'s symbols for ESIs `0..k + repair`, from `data`
/// zero-padded to `k * T`.
fn reference_symbols(
    config: &ObjectTransmissionInformation,
    data: &[u8],
    k: u32,
    repair: u32,
    plan: Option<&SourceBlockEncodingPlan>,
) -> Vec<Vec<u8>> {
    let mut padded = data.to_vec();
    padded.resize(k as usize * config.symbol_size() as usize, 0);
    let encoder = match plan {
        Some(plan) => SourceBlockEncoder::with_encoding_plan(0, config, &padded, plan),
        None => SourceBlockEncoder::new(0, config, &padded),
    };
    let mut out: Vec<Vec<u8>> = encoder
        .source_packets()
        .into_iter()
        .map(|p| p.data().to_vec())
        .collect();
    for (i, p) in encoder.repair_packets(0, repair).into_iter().enumerate() {
        assert_eq!(p.payload_id().encoding_symbol_id(), k + i as u32);
        out.push(p.data().to_vec());
    }
    out
}

fn new_range(encoder: &ReusableSourceBlockEncoder, first: u32, count: u32) -> Vec<Vec<u8>> {
    let t = encoder.symbol_size() as usize;
    let mut out = vec![0u8; count as usize * t];
    encoder
        .repair_range_into(first, count, &mut out, t)
        .unwrap();
    out.chunks(t).map(|c| c.to_vec()).collect()
}

/// Encode `data` as a K-symbol block with every arm and compare all ESIs
/// `0..k + repair` with `SourceBlockEncoder`.
fn check_encode(cfg: &ObjectTransmissionInformation, data: &[u8], k: u32, repair: u32) {
    let t = cfg.symbol_size() as usize;
    let reference = reference_symbols(cfg, data, k, repair, None);
    let plan = Arc::new(SourceBlockEncodingPlan::generate(k as u16));
    assert_eq!(
        reference,
        reference_symbols(cfg, data, k, repair, Some(&plan)),
        "plan and planless SourceBlockEncoder differ at k={k}"
    );

    // Planless arm.
    let mut planless = ReusableSourceBlockEncoder::new(cfg).unwrap();
    planless.encode_block(data, k).unwrap();
    assert_eq!(
        new_range(&planless, 0, k + repair),
        reference,
        "planless k={k}"
    );

    // Planned arm.
    let mut planned = ReusableSourceBlockEncoder::with_plan(cfg, plan).unwrap();
    planned.encode_block(data, k).unwrap();
    assert_eq!(
        new_range(&planned, 0, k + repair),
        reference,
        "planned k={k}"
    );

    // A plan for another size is not used: the block is solved directly.
    let other = Arc::new(SourceBlockEncodingPlan::generate(k as u16 + 1));
    let mut mismatched = ReusableSourceBlockEncoder::with_plan(cfg, other).unwrap();
    mismatched.encode_block(data, k).unwrap();
    assert_eq!(
        new_range(&mismatched, 0, k + repair),
        reference,
        "other plan k={k}"
    );

    // repair_into with an arbitrary ESI list and a header-reserving stride.
    let mut esis: Vec<u32> = (0..k + repair).collect();
    esis.push(k + repair + 1000);
    esis.push(0);
    let mut rng = Rng::new(u64::from(k));
    rng.shuffle(&mut esis);
    let stride = t + 8;
    let mut out = vec![0xA5u8; esis.len() * stride];
    planned.repair_into(&esis, &mut out[8..], stride).unwrap();
    let far = SourceBlockEncoder::new(0, cfg, &{
        let mut p = data.to_vec();
        p.resize(k as usize * t, 0);
        p
    })
    .repair_packets(repair + 1000, 1)[0]
        .data()
        .to_vec();
    for (i, &esi) in esis.iter().enumerate() {
        let got = &out[i * stride + 8..i * stride + 8 + t];
        let want = if esi < k + repair {
            &reference[esi as usize]
        } else {
            &far
        };
        assert_eq!(got, &want[..], "repair_into esi={esi} k={k}");
        assert!(out[i * stride..i * stride + 8].iter().all(|&b| b == 0xA5));
    }
}

#[test]
fn encoder_matches_source_block_encoder_for_every_small_k() {
    let cfg = config(16, 1, 1);
    let mut rng = Rng::new(1);
    for k in 1..=80u32 {
        let full = rng.bytes(k as usize * 16);
        check_encode(&cfg, &full, k, 12);
        // A partial last symbol, and a block that is mostly zero padding.
        check_encode(&cfg, &full[..full.len() - 5], k, 12);
        check_encode(&cfg, &full[..full.len() / 3], k, 4);
    }
}

#[test]
fn encoder_matches_source_block_encoder_at_production_sizes() {
    let mut rng = Rng::new(2);
    for (k, t) in [
        (128u32, 1344u16),
        (12, 1344),
        (16, 1344),
        (51, 1344),
        (5, 1344),
    ] {
        if !FULL && k == 128 {
            continue;
        }
        let cfg = config(t, 1, 8);
        let data = rng.bytes(k as usize * t as usize);
        check_encode(&cfg, &data, k, 26);
        check_encode(&cfg, &data[..data.len() - 10], k, 26);
    }
    // The sparse solver threshold (K' >= 250) and larger blocks.
    let large: &[u32] = if FULL {
        &[249, 250, 251, 300, 1000]
    } else {
        &[250]
    };
    for &k in large {
        let cfg = config(8, 1, 8);
        let data = rng.bytes(k as usize * 8 - 3);
        check_encode(&cfg, &data, k, 20);
    }
}

#[test]
fn encoder_handles_sub_blocks() {
    let mut rng = Rng::new(3);
    for (t, n, al) in [
        (24u16, 2u16, 4u8),
        (24, 3, 4),
        (40, 3, 8),
        (12, 5, 1),
        (8, 9, 1),
    ] {
        let cfg = config(t, n, al);
        for k in [1u32, 2, 7, 20] {
            let data = rng.bytes(k as usize * t as usize);
            check_encode(&cfg, &data, k, 6);
            check_encode(&cfg, &data[..data.len() - 1], k, 6);
        }
    }
}

#[test]
fn encoder_repair_at_the_esi_limit() {
    let cfg = config(8, 1, 8);
    let data = Rng::new(4).bytes(10 * 8);
    let mut encoder = ReusableSourceBlockEncoder::new(&cfg).unwrap();
    encoder.encode_block(&data, 10).unwrap();
    let reference = SourceBlockEncoder::new(0, &cfg, &data);
    let last = (1u32 << 24) - 1;
    let want = reference.repair_packets(last - 10, 1)[0].data().to_vec();
    let mut out = vec![0u8; 8];
    encoder.repair_range_into(last, 1, &mut out, 8).unwrap();
    assert_eq!(out, want);
    assert_eq!(
        encoder.repair_range_into(last, 2, &mut [0u8; 16], 8),
        Err(BlockError::InvalidEsi(1 << 24))
    );
    assert_eq!(
        encoder.repair_into(&[3, 1 << 24], &mut [0u8; 16], 8),
        Err(BlockError::InvalidEsi(1 << 24))
    );
    // count 0 needs no buffer.
    encoder.repair_range_into(0, 0, &mut [], 8).unwrap();
    encoder.repair_into(&[], &mut [], 8).unwrap();
}

/// The short block an interleaved encoder closes early: a plan is attached
/// for the full block, and the K' < K block must come out as its own
/// K'-symbol source block.
#[test]
fn short_block_with_full_block_plan_attached() {
    let cfg = config(1344, 1, 8);
    let mut rng = Rng::new(5);
    let plan = Arc::new(SourceBlockEncodingPlan::generate(128));
    let mut encoder = ReusableSourceBlockEncoder::with_plan(&cfg, plan).unwrap();
    for k in [128u32, 51, 128, 1, 127, 128, 64] {
        let data = rng.bytes(k as usize * 1344);
        encoder.encode_block(&data, k).unwrap();
        assert_eq!(encoder.source_symbols(), k);
        let reference = reference_symbols(&cfg, &data, k, 26, None);
        assert_eq!(new_range(&encoder, 0, k + 26), reference, "k={k}");
    }
}

/// Every source symbol of a K-symbol block, then `repair` repair symbols.
fn block_symbols(
    cfg: &ObjectTransmissionInformation,
    data: &[u8],
    k: u32,
    repair: u32,
) -> Vec<Vec<u8>> {
    let mut encoder = ReusableSourceBlockEncoder::new(cfg).unwrap();
    encoder.encode_block(data, k).unwrap();
    new_range(&encoder, 0, k + repair)
}

/// Decode `received` with `SourceBlockDecoder`, one packet at a time, as the
/// interleaved decoder does.
fn reference_decode(
    cfg: &ObjectTransmissionInformation,
    k: u32,
    received: &[(u32, &[u8])],
) -> Option<Vec<u8>> {
    let mut decoder = SourceBlockDecoder::new(0, cfg, u64::from(k) * u64::from(cfg.symbol_size()));
    let mut result = None;
    for &(esi, data) in received {
        result = decoder.decode([EncodingPacket::new(PayloadId::new(0, esi), data.to_vec())]);
        if result.is_some() {
            break;
        }
    }
    result
}

fn new_decode(
    decoder: &mut ReusableSourceBlockDecoder,
    k: u32,
    received: &[(u32, &[u8])],
) -> Option<Vec<u8>> {
    decoder.reset(k).unwrap();
    for &(esi, data) in received {
        decoder.add_symbol(esi, data).unwrap();
    }
    if !decoder.try_decode().unwrap() {
        return None;
    }
    let mut out = vec![0u8; k as usize * decoder.symbol_size() as usize];
    decoder.copy_block_into(0, &mut out).unwrap();
    Some(out)
}

/// A seeded erasure pattern: drop `lost` source symbols, keep `extra` more
/// repair symbols than needed, shuffle, and sometimes duplicate a symbol.
fn erasure<'a>(
    rng: &mut Rng,
    symbols: &'a [Vec<u8>],
    k: u32,
    lost: u32,
    extra: u32,
) -> Vec<(u32, &'a [u8])> {
    let mut source: Vec<u32> = (0..k).collect();
    rng.shuffle(&mut source);
    let mut keep: Vec<u32> = source[lost.min(k) as usize..].to_vec();
    let repair_needed = lost.min(k) + extra;
    let repair_avail = symbols.len() as u32 - k;
    let mut repair: Vec<u32> = (k..k + repair_avail).collect();
    rng.shuffle(&mut repair);
    keep.extend(repair.into_iter().take(repair_needed as usize));
    rng.shuffle(&mut keep);
    let mut out: Vec<(u32, &[u8])> = keep
        .iter()
        .map(|&e| (e, &symbols[e as usize][..]))
        .collect();
    if !out.is_empty() && rng.below(3) == 0 {
        let dup = out[rng.below(out.len() as u64) as usize];
        out.insert(rng.below(out.len() as u64 + 1) as usize, dup);
    }
    out
}

#[test]
fn decoder_matches_source_block_decoder_over_seeded_erasures() {
    let mut rng = Rng::new(6);
    let mut decoder_cache: Vec<(u16, ReusableSourceBlockDecoder)> = Vec::new();
    let mut outcomes = [0u32; 2];
    let trials = if FULL { 600 } else { 60 };
    for trial in 0..trials {
        let (t, n, al) = match trial % 4 {
            0 => (1344u16, 1u16, 8u8),
            1 => (16, 1, 1),
            2 => (24, 3, 4),
            _ => (64, 1, 8),
        };
        let cfg = config(t, n, al);
        let k = match trial % 5 {
            0 => 128,
            1 => 1 + rng.below(20) as u32,
            2 => 12,
            3 => 16,
            _ => 1 + rng.below(300) as u32,
        };
        let trim = rng.below(t as u64) as usize;
        let data = rng.bytes(k as usize * t as usize - trim);
        let mut full = data.clone();
        full.resize(k as usize * t as usize, 0);
        let symbols = block_symbols(&cfg, &data, k, k.min(40) + 3);
        let lost = rng.below(u64::from(k.min(40)) + 1) as u32;
        let extra = rng.below(3) as u32;
        let received = erasure(&mut rng, &symbols, k, lost, extra);

        let want = reference_decode(&cfg, k, &received);
        // Reuse one decoder per configuration across trials.
        let slot = match decoder_cache
            .iter()
            .position(|(key, _)| *key == trial as u16 % 4)
        {
            Some(i) => i,
            None => {
                decoder_cache.push((
                    trial as u16 % 4,
                    ReusableSourceBlockDecoder::new(&cfg).unwrap(),
                ));
                decoder_cache.len() - 1
            }
        };
        let got = new_decode(&mut decoder_cache[slot].1, k, &received);
        let fresh = new_decode(
            &mut ReusableSourceBlockDecoder::new(&cfg).unwrap(),
            k,
            &received,
        );
        assert_eq!(got, want, "trial {trial} k={k} lost={lost} extra={extra}");
        assert_eq!(fresh, want, "trial {trial}");
        if let Some(block) = &got {
            assert_eq!(block, &full, "trial {trial}");
        }
        outcomes[got.is_some() as usize] += 1;
    }
    // The patterns must exercise both outcomes; failures here are the
    // rank-deficient draws at zero overhead.
    assert!(outcomes[1] > trials * 5 / 6, "{outcomes:?}");
}

#[test]
fn decode_needs_k_symbols_and_skips_work_without_new_ones() {
    let cfg = config(32, 1, 8);
    let k = 20;
    let data = Rng::new(7).bytes(k as usize * 32);
    let symbols = block_symbols(&cfg, &data, k, 10);
    let mut decoder = ReusableSourceBlockDecoder::new(&cfg).unwrap();
    assert_eq!(decoder.try_decode(), Err(BlockError::NoBlock));
    decoder.reset(k).unwrap();
    // Lose source symbols 0 and 1; feed the rest and one repair symbol.
    for esi in 2..k + 1 {
        assert!(decoder.add_symbol(esi, &symbols[esi as usize]).unwrap());
    }
    assert_eq!(decoder.received_symbols(), k - 1);
    assert!(!decoder.try_decode().unwrap());
    assert_eq!(
        decoder.solve_count(),
        0,
        "fewer than K symbols never solves"
    );

    // Duplicates are ignored and do not make the next attempt run.
    assert!(!decoder.add_symbol(5, &symbols[5]).unwrap());
    assert!(!decoder.add_symbol(k, &symbols[k as usize]).unwrap());
    assert!(decoder.has_symbol(k) && !decoder.has_symbol(0));
    assert!(decoder.add_symbol(k + 1, &symbols[k as usize + 1]).unwrap());
    assert!(decoder.try_decode().unwrap());
    assert_eq!(decoder.solve_count(), 1);
    // Decoded: further attempts and adds do nothing.
    assert!(decoder.try_decode().unwrap());
    assert!(!decoder.add_symbol(0, &symbols[0]).unwrap());
    assert_eq!(decoder.solve_count(), 1);
    let mut out = vec![0u8; k as usize * 32];
    decoder.copy_block_into(0, &mut out).unwrap();
    assert_eq!(out, data);
    // A symbol range: symbols 3..7.
    let mut part = vec![0u8; 4 * 32];
    decoder.copy_block_into(3 * 32, &mut part).unwrap();
    assert_eq!(part, data[3 * 32..7 * 32]);

    // All source symbols: decodes without solving.
    decoder.reset(k).unwrap();
    for esi in (0..k).rev() {
        decoder.add_symbol(esi, &symbols[esi as usize]).unwrap();
    }
    assert!(decoder.try_decode().unwrap());
    assert_eq!(decoder.solve_count(), 1);
    decoder.copy_block_into(0, &mut out).unwrap();
    assert_eq!(out, data);
}

/// Exactly K symbols that do not determine the block: the first attempt
/// fails after one solve, a retry with nothing new does not solve again,
/// and one more symbol completes the block.
#[test]
fn rank_deficient_attempt_then_success() {
    let cfg = config(16, 1, 8);
    let k = 10;
    let mut found = false;
    for seed in 0..2000u64 {
        let mut rng = Rng::new(seed);
        let data = rng.bytes(k as usize * 16);
        let symbols = block_symbols(&cfg, &data, k, 30);
        let mut esis: Vec<u32> = (0..k + 30).collect();
        rng.shuffle(&mut esis);
        let mut decoder = ReusableSourceBlockDecoder::new(&cfg).unwrap();
        decoder.reset(k).unwrap();
        for &esi in &esis[..k as usize] {
            decoder.add_symbol(esi, &symbols[esi as usize]).unwrap();
        }
        if decoder.received_source_symbols() == k || decoder.try_decode().unwrap() {
            continue;
        }
        found = true;
        assert_eq!(decoder.solve_count(), 1);
        assert!(!decoder.try_decode().unwrap());
        assert_eq!(decoder.solve_count(), 1, "no new symbol, no new solve");
        let mut next = k as usize;
        while !decoder.try_decode().unwrap() {
            decoder
                .add_symbol(esis[next], &symbols[esis[next] as usize])
                .unwrap();
            next += 1;
        }
        let mut out = vec![0u8; k as usize * 16];
        decoder.copy_block_into(0, &mut out).unwrap();
        assert_eq!(out, data);
        break;
    }
    assert!(found, "no rank-deficient pattern in 2000 seeds");
}

/// One encoder driven through a long seeded sequence of block sizes, with
/// and without the attached plan's size, compared with a fresh encoder and
/// with `SourceBlockEncoder` for every block. Guards the storage reuse
/// against stale state from a previous, larger or reordered block.
#[test]
fn reused_encoder_matches_fresh_encoders() {
    let mut rng = Rng::new(8);
    for (t, n, al) in [(64u16, 1u16, 8u8), (24, 3, 4)] {
        let cfg = config(t, n, al);
        let plan = Arc::new(SourceBlockEncodingPlan::generate(48));
        let mut reused = ReusableSourceBlockEncoder::with_plan(&cfg, plan.clone()).unwrap();
        for step in 0..120 {
            let k = match rng.below(4) {
                0 | 1 => 48,
                2 => 1 + rng.below(48) as u32,
                _ => 1 + rng.below(200) as u32,
            };
            let trim = rng.below(u64::from(t) * 2).min(k as u64 * t as u64 - 1) as usize;
            let len = k as usize * t as usize - trim;
            let data = rng.bytes(len);
            if rng.below(10) == 0 {
                reused.clear();
                assert_eq!(
                    reused.repair_range_into(0, 1, &mut vec![0; t as usize], t as usize),
                    Err(BlockError::NoBlock)
                );
            }
            reused.encode_block(&data, k).unwrap();
            let mut fresh = ReusableSourceBlockEncoder::with_plan(&cfg, plan.clone()).unwrap();
            fresh.encode_block(&data, k).unwrap();
            let r = 1 + rng.below(30) as u32;
            let got = new_range(&reused, 0, k + r);
            assert_eq!(got, new_range(&fresh, 0, k + r), "step {step} k={k}");
            assert_eq!(
                got,
                reference_symbols(&cfg, &data, k, r, None),
                "step {step} k={k}"
            );
        }
    }
}

/// One decoder driven through a long seeded sequence: blocks of different
/// K, failed attempts followed by successful ones, and resets part way
/// through a block. Each completed block is compared with a fresh decoder
/// fed the same symbols, and the observable state with the fresh one's.
#[test]
fn reused_decoder_matches_fresh_decoders() {
    let mut rng = Rng::new(9);
    for (t, n, al) in [(64u16, 1u16, 8u8), (24, 3, 4)] {
        let cfg = config(t, n, al);
        let mut reused = ReusableSourceBlockDecoder::new(&cfg).unwrap();
        for step in 0..150 {
            let k = match rng.below(3) {
                0 => 32,
                1 => 1 + rng.below(32) as u32,
                _ => 1 + rng.below(150) as u32,
            };
            let data = rng.bytes(k as usize * t as usize);
            let symbols = block_symbols(&cfg, &data, k, k + 10);
            let mut order: Vec<u32> = (0..2 * k + 10).collect();
            rng.shuffle(&mut order);

            if rng.below(5) == 0 {
                // Abandon a block part way through: reset mid-block.
                reused.reset(1 + rng.below(100) as u32).unwrap();
                let other = rng.bytes(t as usize);
                for esi in 0..rng.below(20) as u32 {
                    let _ = reused.add_symbol(esi * 3, &other);
                }
                if rng.below(2) == 0 {
                    let _ = reused.try_decode();
                }
            }

            reused.reset(k).unwrap();
            let mut fresh = ReusableSourceBlockDecoder::new(&cfg).unwrap();
            fresh.reset(k).unwrap();
            // Feed in batches, attempting after each, until decoded; first
            // batches are often short of K so attempts fail before one works.
            let mut fed = 0usize;
            let mut decoded = false;
            while !decoded {
                assert!(fed < order.len(), "step {step}: ran out of symbols");
                let batch = 1 + rng.below(u64::from(k) / 2 + 1) as usize;
                for &esi in &order[fed..(fed + batch).min(order.len())] {
                    let a = reused.add_symbol(esi, &symbols[esi as usize]).unwrap();
                    let b = fresh.add_symbol(esi, &symbols[esi as usize]).unwrap();
                    assert_eq!(a, b, "step {step}");
                }
                fed = (fed + batch).min(order.len());
                decoded = reused.try_decode().unwrap();
                assert_eq!(decoded, fresh.try_decode().unwrap(), "step {step}");
                assert_eq!(reused.received_symbols(), fresh.received_symbols());
                assert_eq!(
                    reused.received_source_symbols(),
                    fresh.received_source_symbols()
                );
                assert_eq!(reused.is_decoded(), fresh.is_decoded());
            }
            let mut a = vec![0u8; k as usize * t as usize];
            let mut b = vec![0u8; k as usize * t as usize];
            reused.copy_block_into(0, &mut a).unwrap();
            fresh.copy_block_into(0, &mut b).unwrap();
            assert_eq!(a, b, "step {step}");
            assert_eq!(a, data, "step {step}");
        }
    }
}

#[test]
fn encoder_errors_leave_state_unchanged() {
    assert_eq!(
        ReusableSourceBlockEncoder::new(&ObjectTransmissionInformation::deserialize(&[0; 12])),
        Err(BlockError::InvalidConfig)
    );
    let cfg = config(16, 1, 8);
    let mut encoder =
        ReusableSourceBlockEncoder::with_plan(&cfg, Arc::new(SourceBlockEncodingPlan::generate(8)))
            .unwrap();
    // Before any block.
    let before = encoder.clone();
    assert_eq!(
        encoder.repair_range_into(0, 1, &mut [0u8; 16], 16),
        Err(BlockError::NoBlock)
    );
    assert_eq!(
        encoder.repair_into(&[0], &mut [0u8; 16], 16),
        Err(BlockError::NoBlock)
    );
    assert_eq!(encoder, before);

    let data = Rng::new(10).bytes(8 * 16);
    encoder.encode_block(&data, 8).unwrap();
    let before = encoder.clone();
    assert_eq!(
        encoder.encode_block(&data, 0),
        Err(BlockError::InvalidSourceSymbols(0))
    );
    assert_eq!(
        encoder.encode_block(&data, 56404),
        Err(BlockError::InvalidSourceSymbols(56404))
    );
    assert_eq!(
        encoder.encode_block(&data, 7),
        Err(BlockError::DataTooLong { len: 128, max: 112 })
    );
    assert_eq!(
        encoder.encode_block(&[0u8; 8 * 16 + 1], 8),
        Err(BlockError::DataTooLong { len: 129, max: 128 })
    );
    assert_eq!(encoder.reserve(0), Err(BlockError::InvalidSourceSymbols(0)));
    assert_eq!(encoder, before);

    let mut out = vec![0xA5u8; 40];
    assert_eq!(
        encoder.repair_range_into(8, 3, &mut out, 15),
        Err(BlockError::InvalidStride {
            stride: 15,
            symbol_size: 16
        })
    );
    assert_eq!(
        encoder.repair_range_into(8, 3, &mut out, 16),
        Err(BlockError::BufferTooSmall {
            len: 40,
            needed: 48
        })
    );
    assert_eq!(
        encoder.repair_into(&[8, 9, 10], &mut out, 16),
        Err(BlockError::BufferTooSmall {
            len: 40,
            needed: 48
        })
    );
    assert_eq!(
        encoder.repair_into(&[8, u32::MAX], &mut out, 16),
        Err(BlockError::InvalidEsi(u32::MAX))
    );
    assert!(out.iter().all(|&b| b == 0xA5), "nothing written on error");
    assert_eq!(encoder, before);
    // The block is still the one loaded before the errors.
    assert_eq!(
        new_range(&encoder, 0, 12),
        reference_symbols(&cfg, &data, 8, 4, None)
    );
}

#[test]
fn decoder_errors_leave_state_unchanged() {
    let cfg = config(16, 1, 8);
    let mut decoder = ReusableSourceBlockDecoder::new(&cfg).unwrap();
    let before = decoder.clone();
    assert_eq!(decoder.add_symbol(0, &[0u8; 16]), Err(BlockError::NoBlock));
    assert_eq!(
        decoder.copy_block_into(0, &mut [0u8; 16]),
        Err(BlockError::NoBlock)
    );
    assert_eq!(decoder.reset(0), Err(BlockError::InvalidSourceSymbols(0)));
    assert_eq!(decoder, before);

    let k = 6;
    let data = Rng::new(11).bytes(k as usize * 16);
    let symbols = block_symbols(&cfg, &data, k, 4);
    decoder.reset(k).unwrap();
    for esi in [0, 2, 3, 6] {
        decoder.add_symbol(esi, &symbols[esi as usize]).unwrap();
    }
    let before = decoder.clone();
    assert_eq!(
        decoder.add_symbol(1, &[0u8; 15]),
        Err(BlockError::SymbolSizeMismatch {
            len: 15,
            expected: 16
        })
    );
    assert_eq!(
        decoder.add_symbol(1 << 24, &[0u8; 16]),
        Err(BlockError::InvalidEsi(1 << 24))
    );
    assert_eq!(
        decoder.reset(56404),
        Err(BlockError::InvalidSourceSymbols(56404))
    );
    assert_eq!(
        decoder.reserve(0, 1),
        Err(BlockError::InvalidSourceSymbols(0))
    );
    let mut out = vec![0xA5u8; 16];
    assert_eq!(
        decoder.copy_block_into(0, &mut out),
        Err(BlockError::NotDecoded)
    );
    assert_eq!(decoder, before);

    for esi in [7, 8] {
        decoder.add_symbol(esi, &symbols[esi as usize]).unwrap();
    }
    assert!(decoder.try_decode().unwrap());
    assert_eq!(
        decoder.copy_block_into(90, &mut out),
        Err(BlockError::OutOfRange {
            offset: 90,
            len: 16,
            block_len: 96
        })
    );
    assert_eq!(
        decoder.copy_block_into(usize::MAX, &mut out),
        Err(BlockError::OutOfRange {
            offset: usize::MAX,
            len: 16,
            block_len: 96
        })
    );
    assert!(out.iter().all(|&b| b == 0xA5), "nothing written on error");
    decoder.copy_block_into(80, &mut out).unwrap();
    assert_eq!(out, data[80..]);
}

#[cfg(feature = "std")]
#[test]
fn cached_plan_is_shared_and_equal_to_a_generated_one() {
    let a = SourceBlockEncodingPlan::cached(37);
    let b = SourceBlockEncodingPlan::cached(37);
    assert!(Arc::ptr_eq(&a, &b));
    assert_eq!(*a, SourceBlockEncodingPlan::generate(37));
    assert_eq!(a.source_symbol_count(), 37);
}
