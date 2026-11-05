//! `ObjectEncoder` against the classic `Encoder`, byte for byte: source and
//! repair packets, over source blocks (Z), sub-blocks (N), unaligned
//! transfer lengths (F) and block sizes on both sides of the plan store's
//! K' limit.

use std::mem::MaybeUninit;

use raptorq::{Encoder, EncodingPacket, ObjectEncoder, ObjectTransmissionInformation, PayloadId};

const T: u16 = 1312;
const AL: u8 = 8;
const REPAIR: u32 = 6;

/// The whole matrix runs only optimized (CI's release runs); unoptimized
/// runs, `cargo test --no-default-features` among them, take a subset.
const FULL: bool = !cfg!(debug_assertions);

fn data(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Every packet of block `sbn`: its source packets, then repair symbols
/// `first_repair..first_repair + REPAIR`.
fn object_packets(encoder: &ObjectEncoder, first_repair: u32) -> Vec<EncodingPacket> {
    let t = usize::from(encoder.config().symbol_size());
    let mut packets = vec![];
    for (sbn, block) in encoder.blocks().iter().enumerate() {
        let k = block.closed_source_symbols().unwrap();
        for (first, count) in [(0, k), (k + first_repair, REPAIR)] {
            let mut out = vec![MaybeUninit::<u8>::uninit(); count as usize * t];
            block.symbols_into(first, count, &mut out, t).unwrap();
            for (i, symbol) in out.chunks_exact(t).enumerate() {
                // SAFETY: symbols_into initialized every byte of `out`.
                let bytes = symbol.iter().map(|b| unsafe { b.assume_init() }).collect();
                packets.push(EncodingPacket::new(
                    PayloadId::new(sbn as u8, first + i as u32),
                    bytes,
                ));
            }
        }
    }
    packets
}

fn classic_packets(encoder: &Encoder, first_repair: u32) -> Vec<EncodingPacket> {
    let mut packets = vec![];
    for block in encoder.get_block_encoders() {
        packets.extend(block.source_packets());
        packets.extend(block.repair_packets(first_repair, REPAIR));
    }
    packets
}

fn check(k: u32, z: u8, n: u16, short: usize) {
    let total = k as usize * usize::from(z) * usize::from(T);
    let len = total - short;
    let bytes = data(
        len,
        u64::from(k) << 16 | u64::from(z) << 8 | u64::from(n) | (short as u64) << 32,
    );
    let config = ObjectTransmissionInformation::new(len as u64, T, z, n, AL);
    let classic = Encoder::new(&bytes, config);
    let object = ObjectEncoder::new(&bytes, config).unwrap();
    assert_eq!(object.config(), config);
    assert_eq!(object.blocks().len(), usize::from(z));
    assert_eq!(object.source_symbol_count(), k * u32::from(z));
    for first_repair in [0, 3] {
        assert!(
            classic_packets(&classic, first_repair) == object_packets(&object, first_repair),
            "k={k} Z={z} N={n} F=kT-{short} first_repair={first_repair}"
        );
    }
}

/// The pads of the spec: aligned, 1 byte short, 100 bytes short and T - 8
/// short (more than T / N at N 2 and 4, so it spans the trailing sub-symbols
/// of several symbols).
fn shorts(k: u32, z: u8) -> Vec<usize> {
    let total = k as usize * usize::from(z) * usize::from(T);
    [0, 1, 100, usize::from(T) - 8]
        .into_iter()
        .filter(|&s| s < total)
        .collect()
}

#[test]
fn matches_encoder_over_blocks_sub_blocks_and_pads() {
    let ks: &[u32] = if FULL {
        &[1, 2, 31, 32, 51, 64, 97, 104, 128, 255, 256]
    } else {
        &[1, 2, 31, 51]
    };
    for &k in ks {
        for z in [1, 2, 3] {
            for n in [1, 2, 4] {
                for short in shorts(k, z) {
                    check(k, z, n, short);
                }
            }
        }
    }
}

/// K' > 257: the encoder generates and keeps its own plans.
#[test]
fn matches_encoder_above_the_plan_store() {
    let cases: &[(u32, u8, u16)] = if FULL {
        &[
            (300, 1, 1),
            (300, 2, 2),
            (300, 3, 4),
            (1000, 1, 1),
            (1000, 2, 2),
            (1000, 3, 4),
        ]
    } else {
        &[(300, 1, 2)]
    };
    for &(k, z, n) in cases {
        {
            for short in shorts(k, z) {
                check(k, z, n, short);
            }
        }
    }
}

/// Blocks of unequal size: Kt not a multiple of Z, so the object has long
/// and short blocks with different K'.
#[test]
fn matches_encoder_with_long_and_short_blocks() {
    let cases: &[(usize, u8, u16)] = if FULL {
        &[(65, 2, 1), (100, 3, 2), (517, 2, 4), (7, 3, 1)]
    } else {
        &[(65, 2, 1), (7, 3, 1)]
    };
    for &(kt, z, n) in cases {
        let len = kt * usize::from(T) - 37;
        let bytes = data(len, kt as u64);
        let config = ObjectTransmissionInformation::new(len as u64, T, z, n, AL);
        let classic = Encoder::new(&bytes, config);
        let object = ObjectEncoder::new(&bytes, config).unwrap();
        assert!(
            classic_packets(&classic, 0) == object_packets(&object, 0),
            "Kt={kt} Z={z}"
        );
    }
}

#[test]
fn rejects_an_empty_object() {
    let config = ObjectTransmissionInformation::new(0, T, 1, 1, AL);
    assert!(ObjectEncoder::new(&[], config).is_err());
}
