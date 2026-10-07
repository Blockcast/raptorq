// WebAssembly SIMD128 GF(2^8) kernels.
//
// wasm32 has no runtime CPU feature detection: SIMD128 is either compiled in
// (`-C target-feature=+simd128`) or absent. This module is therefore compiled
// only when the feature is statically enabled, and `octets.rs` dispatches to it
// unconditionally on that target. Every kernel produces exactly the bytes the
// scalar fallbacks in `octets.rs` produce; the tails that do not fill a 16-byte
// vector are handed to those fallbacks.
//
// Originally written for Blockcast fec-raptorq (kkroo/fec-raptorq 5842ded,
// cfg-gated in 3433992) against raptorq 2.0.0, and ported onto the upstream
// structure that split per-ISA kernels into their own modules (lib.rs gates
// this module on wasm32 + simd128 + std).

use core::arch::wasm32::*;

use crate::octet::{OCTET_MUL_HI_BITS, OCTET_MUL_LOW_BITS, Octet};
use crate::octets::{
    BinaryOctetVec, add_assign_fallback, fused_addassign_mul_scalar_fallback,
    mulassign_scalar_fallback,
};

const LANES: usize = 16;

#[inline(always)]
fn mul_tables(scalar: &Octet) -> (v128, v128) {
    let s = usize::from(scalar.byte());
    // SAFETY: each table row is [u8; 32]; v128_load reads its first 16 bytes and
    // has no alignment requirement.
    unsafe {
        (
            v128_load(OCTET_MUL_LOW_BITS[s].as_ptr() as *const v128),
            v128_load(OCTET_MUL_HI_BITS[s].as_ptr() as *const v128),
        )
    }
}

// Nibble-table multiply of all 16 lanes of `v` by the scalar the tables encode
// (Plank et al., "Screaming Fast Galois Field Arithmetic Using Intel SIMD
// Instructions"). u8x16_shr is a logical shift, so the high nibble needs no mask.
#[inline(always)]
fn mul_vec(v: v128, low_table: v128, hi_table: v128) -> v128 {
    let low = u8x16_swizzle(low_table, v128_and(v, u8x16_splat(0x0F)));
    let hi = u8x16_swizzle(hi_table, u8x16_shr(v, 4));
    v128_xor(low, hi)
}

pub(crate) fn mulassign_scalar_simd128(octets: &mut [u8], scalar: &Octet) {
    let (low_table, hi_table) = mul_tables(scalar);
    let mut chunks = octets.chunks_exact_mut(LANES);
    for chunk in &mut chunks {
        let p = chunk.as_mut_ptr() as *mut v128;
        // SAFETY: `chunk` is exactly 16 bytes; v128 loads/stores are unaligned.
        unsafe { v128_store(p, mul_vec(v128_load(p), low_table, hi_table)) };
    }
    mulassign_scalar_fallback(chunks.into_remainder(), scalar);
}

pub(crate) fn fused_addassign_mul_scalar_simd128(octets: &mut [u8], other: &[u8], scalar: &Octet) {
    assert_eq!(octets.len(), other.len());
    let (low_table, hi_table) = mul_tables(scalar);
    let mut chunks = octets.chunks_exact_mut(LANES);
    let mut other_chunks = other.chunks_exact(LANES);
    for (chunk, other_chunk) in (&mut chunks).zip(&mut other_chunks) {
        let p = chunk.as_mut_ptr() as *mut v128;
        // SAFETY: both chunks are exactly 16 bytes; loads/stores are unaligned.
        unsafe {
            let product = mul_vec(
                v128_load(other_chunk.as_ptr() as *const v128),
                low_table,
                hi_table,
            );
            v128_store(p, v128_xor(v128_load(p), product));
        }
    }
    fused_addassign_mul_scalar_fallback(chunks.into_remainder(), other_chunks.remainder(), scalar);
}

pub(crate) fn add_assign_simd128(octets: &mut [u8], other: &[u8]) {
    assert_eq!(octets.len(), other.len());
    let mut chunks = octets.chunks_exact_mut(LANES);
    let mut other_chunks = other.chunks_exact(LANES);
    for (chunk, other_chunk) in (&mut chunks).zip(&mut other_chunks) {
        let p = chunk.as_mut_ptr() as *mut v128;
        // SAFETY: both chunks are exactly 16 bytes; loads/stores are unaligned.
        unsafe {
            let other_vec = v128_load(other_chunk.as_ptr() as *const v128);
            v128_store(p, v128_xor(v128_load(p), other_vec));
        }
    }
    add_assign_fallback(chunks.into_remainder(), other_chunks.remainder());
}

// octets[i] ^= scalar * bit(padding + i), where `other` is bit-packed into u64
// words starting at bit `padding_bits()` of the first word (see BinaryOctetVec).
// The caller guarantees `octets.len() == other.len()` and a non-empty input.
pub(crate) fn fused_addassign_mul_scalar_binary_simd128(
    octets: &mut [u8],
    other: &BinaryOctetVec,
    scalar: &Octet,
) {
    let words = other.elements();
    let first_bit = other.padding_bits();
    let mut words_iter = words.iter();
    let mut start = 0;
    // The first word holds only 64 - padding live bits; handle them bytewise so
    // every later word lines up with a whole 64-byte stretch of `octets`.
    if first_bit > 0 {
        let first = *words_iter.next().expect("non-empty BinaryOctetVec");
        start = BinaryOctetVec::WORD_WIDTH - first_bit;
        for (i, val) in octets[..start].iter_mut().enumerate() {
            // The bit is 0 or 1, so u8 multiplication is GF(256) multiplication.
            *val ^= scalar.byte() * (((first >> (first_bit + i)) & 1) as u8);
        }
    }
    debug_assert_eq!(
        (octets.len() - start) % BinaryOctetVec::WORD_WIDTH,
        0,
        "BinaryOctetVec words must cover octets exactly"
    );

    let scalar_vec = u8x16_splat(scalar.byte());
    // Byte k of a 16-lane group takes bit k of a u16: lanes 0..8 read the low
    // byte, lanes 8..16 the high byte, each tested against its own bit.
    let spread = u8x16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    let bit_select = u8x16(
        1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80, 1, 2, 4, 8, 0x10, 0x20, 0x40, 0x80,
    );
    for (chunk, &bits) in octets[start..]
        .chunks_exact_mut(BinaryOctetVec::WORD_WIDTH)
        .zip(words_iter)
    {
        // Constraint-matrix rows are sparse: most words are zero.
        if bits == 0 {
            continue;
        }
        for (q, lane_group) in chunk.chunks_exact_mut(LANES).enumerate() {
            let half = (bits >> (16 * q)) as u16;
            if half == 0 {
                continue;
            }
            let expanded = u8x16_swizzle(u16x8_splat(half), spread);
            let mask = u8x16_eq(v128_and(expanded, bit_select), bit_select);
            let product = v128_and(mask, scalar_vec);
            let p = lane_group.as_mut_ptr() as *mut v128;
            // SAFETY: `lane_group` is exactly 16 bytes; loads/stores are unaligned.
            unsafe { v128_store(p, v128_xor(v128_load(p), product)) };
        }
    }
}
