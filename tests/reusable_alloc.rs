//! Steady-state allocation tests for the reusable block codec: after a
//! warm-up that grows the storage, each operation must not allocate.
//!
//! A counting global allocator counts the allocations (alloc, alloc_zeroed,
//! realloc) of the measuring thread only, so tests running in parallel do
//! not disturb each other. Tests marked `#[ignore = "zero-alloc internals
//! pending"]` cover the paths that still run the allocating RFC 6330 solver
//! (`gen_intermediate_symbols` without a plan, and the decoder's solve);
//! they are the targets for the zero-allocation internals.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;

use raptorq::{
    ObjectTransmissionInformation, ReusableSourceBlockDecoder, ReusableSourceBlockEncoder,
    SourceBlockEncodingPlan,
};

struct Counting;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<u64> = const { Cell::new(0) };
}

fn note() {
    let _ = COUNTING.try_with(|on| {
        if on.get() {
            ALLOCS.with(|n| n.set(n.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Run `op` a few times to warm up, then return the allocations of the next
/// runs. `op` gets the iteration number so it can vary its input.
fn steady_state_allocs(mut op: impl FnMut(usize)) -> u64 {
    for i in 0..3 {
        op(i);
    }
    ALLOCS.with(|n| n.set(0));
    COUNTING.with(|c| c.set(true));
    for i in 3..13 {
        op(i);
    }
    COUNTING.with(|c| c.set(false));
    ALLOCS.with(|n| n.get())
}

/// Production geometry (K=128, T=1344) and the small blocks.
const GEOMETRIES: [(u32, u16, u32); 3] = [(128, 1344, 26), (12, 1344, 3), (16, 1344, 4)];

fn config(t: u16) -> ObjectTransmissionInformation {
    ObjectTransmissionInformation::new(0, t, 1, 1, 8)
}

fn block(seed: usize, len: usize) -> Vec<u8> {
    let mut x = (seed as u32).wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect()
}

fn planned_encoder(k: u32, t: u16) -> ReusableSourceBlockEncoder {
    let plan = Arc::new(SourceBlockEncodingPlan::generate(k as u16));
    ReusableSourceBlockEncoder::with_plan(&config(t), plan).unwrap()
}

#[test]
fn encode_block_with_plan_does_not_allocate() {
    for (k, t, _) in GEOMETRIES {
        let blocks: Vec<Vec<u8>> = (0..13)
            .map(|i| block(i, k as usize * t as usize - i))
            .collect();
        let mut encoder = planned_encoder(k, t);
        let n = steady_state_allocs(|i| encoder.encode_block(&blocks[i], k).unwrap());
        assert_eq!(n, 0, "k={k}");
    }
}

#[test]
fn reserved_encoder_does_not_allocate_on_its_first_block() {
    let (k, t) = (128, 1344);
    let data = block(1, k as usize * t as usize);
    let mut encoder = planned_encoder(k, t);
    encoder.reserve(k).unwrap();
    // The plan's reorder mapping is the only buffer reserve() cannot size
    // exactly; it is at most L entries, which reserve() covers.
    COUNTING.with(|c| c.set(true));
    ALLOCS.with(|n| n.set(0));
    encoder.encode_block(&data, k).unwrap();
    COUNTING.with(|c| c.set(false));
    assert_eq!(ALLOCS.with(|n| n.get()), 0);
}

#[test]
#[ignore = "zero-alloc internals pending"]
fn encode_block_without_plan_does_not_allocate() {
    for (k, t, _) in GEOMETRIES {
        let blocks: Vec<Vec<u8>> = (0..13).map(|i| block(i, k as usize * t as usize)).collect();
        let mut encoder = ReusableSourceBlockEncoder::new(&config(t)).unwrap();
        let n = steady_state_allocs(|i| encoder.encode_block(&blocks[i], k).unwrap());
        assert_eq!(n, 0, "k={k}");
    }
}

/// A short block (K' < K) with the full block's plan attached: solved
/// without a plan.
#[test]
#[ignore = "zero-alloc internals pending"]
fn encode_short_block_does_not_allocate() {
    for (k, t, _) in GEOMETRIES {
        let short = k / 2 - 1;
        let blocks: Vec<Vec<u8>> = (0..13)
            .map(|i| block(i, short as usize * t as usize - 7))
            .collect();
        let mut encoder = planned_encoder(k, t);
        let n = steady_state_allocs(|i| encoder.encode_block(&blocks[i], short).unwrap());
        assert_eq!(n, 0, "k={k} short={short}");
    }
}

#[test]
fn repair_range_into_does_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let mut encoder = planned_encoder(k, t);
        encoder
            .encode_block(&block(1, k as usize * t as usize), k)
            .unwrap();
        let stride = 8 + t as usize;
        let mut out = vec![0u8; r as usize * stride];
        let n = steady_state_allocs(|i| {
            encoder
                .repair_range_into(k + i as u32, r, &mut out[8..], stride)
                .unwrap()
        });
        assert_eq!(n, 0, "k={k}");
    }
}

#[test]
fn repair_into_does_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let mut encoder = planned_encoder(k, t);
        encoder
            .encode_block(&block(2, k as usize * t as usize), k)
            .unwrap();
        let esis: Vec<u32> = (0..r)
            .map(|i| k + 3 * i)
            .chain([0, k - 1, (1 << 24) - 1])
            .collect();
        let mut out = vec![0u8; esis.len() * t as usize];
        let n = steady_state_allocs(|_| encoder.repair_into(&esis, &mut out, t as usize).unwrap());
        assert_eq!(n, 0, "k={k}");
    }
}

/// Every symbol of a K-symbol block followed by `r` repair symbols.
fn symbols(k: u32, t: u16, r: u32, seed: usize) -> Vec<Vec<u8>> {
    let mut encoder = planned_encoder(k, t);
    encoder
        .encode_block(&block(seed, k as usize * t as usize), k)
        .unwrap();
    let mut out = vec![0u8; (k + r) as usize * t as usize];
    encoder
        .repair_range_into(0, k + r, &mut out, t as usize)
        .unwrap();
    out.chunks(t as usize).map(|c| c.to_vec()).collect()
}

#[test]
fn decoder_reset_and_add_symbol_do_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let syms = symbols(k, t, r, 3);
        let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
        // Source symbols and every repair symbol: the repair storage grows
        // during warm-up and is reused after.
        let n = steady_state_allocs(|_| {
            decoder.reset(k).unwrap();
            for (esi, s) in syms.iter().enumerate().skip(1) {
                decoder.add_symbol(esi as u32, s).unwrap();
            }
        });
        assert_eq!(n, 0, "k={k}");
    }
}

#[test]
fn reserved_decoder_does_not_allocate_on_its_first_block() {
    let (k, t, r) = GEOMETRIES[0];
    let syms = symbols(k, t, r, 4);
    let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
    decoder.reserve(k, r).unwrap();
    COUNTING.with(|c| c.set(true));
    ALLOCS.with(|n| n.set(0));
    decoder.reset(k).unwrap();
    for (esi, s) in syms.iter().enumerate().skip(1) {
        decoder.add_symbol(esi as u32, s).unwrap();
    }
    COUNTING.with(|c| c.set(false));
    assert_eq!(ALLOCS.with(|n| n.get()), 0);
}

/// A whole block received without loss: reset, K adds, decode, copy out.
#[test]
fn decode_without_loss_does_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let syms = symbols(k, t, r, 5);
        let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
        let mut out = vec![0u8; k as usize * t as usize];
        let n = steady_state_allocs(|_| {
            decoder.reset(k).unwrap();
            for (esi, s) in syms.iter().enumerate().take(k as usize) {
                decoder.add_symbol(esi as u32, s).unwrap();
            }
            assert!(decoder.try_decode().unwrap());
            decoder.copy_block_into(0, &mut out).unwrap();
        });
        assert_eq!(n, 0, "k={k}");
        assert_eq!(decoder.solve_count(), 0);
    }
}

/// Re-feeding symbols the decoder already has, then attempting again: the
/// rank-deficient re-trigger. Nothing new arrived, so nothing runs.
#[test]
fn duplicate_only_attempt_does_not_allocate_or_solve() {
    for (k, t, r) in GEOMETRIES {
        let syms = symbols(k, t, r, 6);
        let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
        decoder.reset(k).unwrap();
        // K - 1 symbols: not enough, one source symbol missing.
        for (esi, s) in syms.iter().enumerate().skip(1).take(k as usize - 1) {
            decoder.add_symbol(esi as u32, s).unwrap();
        }
        assert!(!decoder.try_decode().unwrap());
        let solves = decoder.solve_count();
        let n = steady_state_allocs(|_| {
            for (esi, s) in syms.iter().enumerate().skip(1).take(k as usize - 1) {
                assert!(!decoder.add_symbol(esi as u32, s).unwrap());
            }
            assert!(!decoder.try_decode().unwrap());
        });
        assert_eq!(n, 0, "k={k}");
        assert_eq!(decoder.solve_count(), solves);
    }
}

#[test]
fn copy_block_into_does_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let syms = symbols(k, t, r, 7);
        let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
        decoder.reset(k).unwrap();
        for (esi, s) in syms.iter().enumerate().skip(1) {
            decoder.add_symbol(esi as u32, s).unwrap();
        }
        assert!(decoder.try_decode().unwrap());
        let mut out = vec![0u8; k as usize * t as usize];
        let n = steady_state_allocs(|i| {
            decoder.copy_block_into(0, &mut out).unwrap();
            let first = i % k as usize;
            let len = (k as usize - first) * t as usize;
            decoder
                .copy_block_into(first * t as usize, &mut out[..len])
                .unwrap();
        });
        assert_eq!(n, 0, "k={k}");
    }
}

/// A block with one lost source symbol recovered from one repair symbol:
/// reset, K adds, solve, copy out.
#[test]
#[ignore = "zero-alloc internals pending"]
fn decode_with_loss_does_not_allocate() {
    for (k, t, r) in GEOMETRIES {
        let syms = symbols(k, t, r, 8);
        let mut decoder = ReusableSourceBlockDecoder::new(&config(t)).unwrap();
        let mut out = vec![0u8; k as usize * t as usize];
        let n = steady_state_allocs(|i| {
            let lost = i % k as usize;
            decoder.reset(k).unwrap();
            for (esi, s) in syms.iter().enumerate().take(k as usize + 1) {
                if esi != lost {
                    decoder.add_symbol(esi as u32, s).unwrap();
                }
            }
            assert!(decoder.try_decode().unwrap());
            decoder.copy_block_into(0, &mut out).unwrap();
        });
        assert_eq!(n, 0, "k={k}");
    }
}
