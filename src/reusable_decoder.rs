#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::base::ObjectTransmissionInformation;
use crate::base::intermediate_tuple;
use crate::constraint_matrix::{
    enc_indices, generate_constraint_matrix, generate_constraint_matrix_no_hdpc,
};
use crate::encoder::SPARSE_MATRIX_THRESHOLD;
use crate::matrix::DenseBinaryMatrix;
use crate::pi_solver::{fused_inverse_mul_symbols, fused_inverse_mul_symbols_no_hdpc};
use crate::reusable::{BlockError, ESI_LIMIT, SubBlockLayout, check_source_symbols};
use crate::sparse_matrix::SparseBinaryMatrix;
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_intermediate_symbols,
    num_ldpc_symbols, num_lt_symbols, num_pi_symbols, systematic_index,
};

/// An RFC 6330 source block decoder that is reset to one source block after
/// another and keeps its storage between them.
///
/// It recovers the same block as [`SourceBlockDecoder`](crate::SourceBlockDecoder)
/// from the same symbols, but the work is split so a caller can batch it:
///
/// - [`add_symbol`](Self::add_symbol) copies one received symbol into the
///   decoder's own storage and ignores a symbol it already has;
/// - [`try_decode`](Self::try_decode) runs the decoder once over everything
///   added, and does no work at all when nothing new arrived since its last
///   attempt;
/// - [`copy_block_into`](Self::copy_block_into) copies the recovered block,
///   or any byte range of it, into a caller buffer.
///
/// Once the storage has grown to the largest block (and the most repair
/// symbols) it sees, resetting, adding symbols, a decode that needs no
/// repair, and copying out do not allocate.
///
/// ```
/// use raptorq::{ObjectTransmissionInformation, ReusableSourceBlockDecoder,
///               ReusableSourceBlockEncoder};
///
/// let config = ObjectTransmissionInformation::new(0, 8, 1, 1, 8);
/// let mut encoder = ReusableSourceBlockEncoder::new(&config).unwrap();
/// encoder.encode_block(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 2).unwrap();
/// let mut symbol = [0u8; 8];
///
/// let mut decoder = ReusableSourceBlockDecoder::new(&config).unwrap();
/// decoder.reset(2).unwrap();
/// // Source symbol 0 is lost; source symbol 1 and repair symbol ESI 2 arrive.
/// for esi in [1, 2] {
///     encoder.repair_into(&[esi], &mut symbol, 8).unwrap();
///     decoder.add_symbol(esi, &symbol).unwrap();
/// }
/// assert!(decoder.try_decode().unwrap());
/// let mut block = [0u8; 16];
/// decoder.copy_block_into(0, &mut block).unwrap();
/// assert_eq!(block[0], 1);
/// ```
#[derive(Clone, Debug)]
pub struct ReusableSourceBlockDecoder {
    layout: SubBlockLayout,
    /// K of the current block; 0 before the first reset.
    source_symbols: u32,
    /// K symbols. Slot `i` holds source symbol `i` once it was received or
    /// the block was decoded.
    source: SymbolSlab,
    /// One bit per source symbol: received.
    received_source: Vec<u64>,
    received_source_count: u32,
    /// Received repair symbols, in arrival order, and their ESIs.
    repair: SymbolSlab,
    repair_esis: Vec<u32>,
    decoded: bool,
    /// A symbol was added since the last decode attempt.
    unattempted: bool,
    solves: u64,
}

/// Two decoders are equal when they hold the same block: the same layout, K,
/// received source and repair symbols, and decode state. The lifetime solve
/// count and the "symbol added since the last attempt" flag are not
/// compared. The flag only skips re-running a solve that already failed with
/// the same symbols, so it cannot change any result.
impl PartialEq for ReusableSourceBlockDecoder {
    fn eq(&self, other: &Self) -> bool {
        self.layout == other.layout
            && self.source_symbols == other.source_symbols
            && self.source == other.source
            && self.received_source == other.received_source
            && self.received_source_count == other.received_source_count
            && self.repair == other.repair
            && self.repair_esis == other.repair_esis
            && self.decoded == other.decoded
    }
}

impl Eq for ReusableSourceBlockDecoder {}

impl ReusableSourceBlockDecoder {
    /// Create a decoder with no block. Call [`reset`](Self::reset) before
    /// adding symbols.
    ///
    /// Only the symbol size, sub-block count and symbol alignment of `config`
    /// are used; each block's size is given to [`reset`](Self::reset).
    ///
    /// # Errors
    /// [`BlockError::InvalidConfig`] if the symbol size, alignment or
    /// sub-block count is zero, or the symbol size is not a multiple of the
    /// alignment.
    pub fn new(config: &ObjectTransmissionInformation) -> Result<Self, BlockError> {
        let layout = SubBlockLayout::new(config)?;
        Ok(ReusableSourceBlockDecoder {
            source_symbols: 0,
            source: SymbolSlab::with_zeros(0, layout.symbol_size()),
            received_source: Vec::new(),
            received_source_count: 0,
            repair: SymbolSlab::with_zeros(0, layout.symbol_size()),
            repair_esis: Vec::new(),
            decoded: false,
            unattempted: false,
            solves: 0,
            layout,
        })
    }

    /// Grow the storage now for a block of `source_symbols` symbols and
    /// `repair_symbols` received repair symbols, so the first such block
    /// does not allocate in [`reset`](Self::reset) or
    /// [`add_symbol`](Self::add_symbol).
    ///
    /// # Errors
    /// [`BlockError::InvalidSourceSymbols`] if `source_symbols` is zero or
    /// above `MAX_SOURCE_SYMBOLS_PER_BLOCK`.
    pub fn reserve(&mut self, source_symbols: u32, repair_symbols: u32) -> Result<(), BlockError> {
        check_source_symbols(source_symbols)?;
        self.source.reserve_symbols(source_symbols as usize);
        let words = (source_symbols as usize).div_ceil(64);
        if words > self.received_source.len() {
            self.received_source
                .reserve(words - self.received_source.len());
        }
        self.repair.reserve_symbols(repair_symbols as usize);
        if (repair_symbols as usize) > self.repair_esis.len() {
            self.repair_esis
                .reserve(repair_symbols as usize - self.repair_esis.len());
        }
        Ok(())
    }

    /// Start a new block of `source_symbols` (K) symbols, discarding every
    /// symbol of the current block and its decoded state. The storage is
    /// kept. Resetting is allowed at any point, including part way through
    /// a block.
    ///
    /// # Errors
    /// [`BlockError::InvalidSourceSymbols`] if K is zero or above
    /// `MAX_SOURCE_SYMBOLS_PER_BLOCK`. The decoder is then unchanged.
    pub fn reset(&mut self, source_symbols: u32) -> Result<(), BlockError> {
        check_source_symbols(source_symbols)?;
        let k = source_symbols as usize;
        self.source_symbols = source_symbols;
        self.source.reset_zeroed(k, self.layout.symbol_size());
        self.received_source.clear();
        self.received_source.resize(k.div_ceil(64), 0);
        self.received_source_count = 0;
        self.repair.reset_zeroed(0, self.layout.symbol_size());
        self.repair_esis.clear();
        self.decoded = false;
        self.unattempted = false;
        Ok(())
    }

    /// The symbol size T in bytes.
    pub fn symbol_size(&self) -> u16 {
        self.layout.symbol_size() as u16
    }

    /// K of the current block, or 0 before the first [`reset`](Self::reset).
    pub fn source_symbols(&self) -> u32 {
        self.source_symbols
    }

    /// The number of distinct symbols (source and repair) added to the
    /// current block.
    pub fn received_symbols(&self) -> u32 {
        self.received_source_count + self.repair_esis.len() as u32
    }

    /// The number of distinct source symbols (ESI below K) added to the
    /// current block.
    pub fn received_source_symbols(&self) -> u32 {
        self.received_source_count
    }

    /// Whether the current block has been recovered.
    pub fn is_decoded(&self) -> bool {
        self.decoded
    }

    /// The number of times this decoder has run the RFC 6330 linear solve,
    /// over its whole lifetime. A decode from source symbols alone does not
    /// solve, and [`try_decode`](Self::try_decode) does not solve again
    /// until a new symbol arrives.
    pub fn solve_count(&self) -> u64 {
        self.solves
    }

    /// Whether symbol `esi` of the current block has been added (or, once
    /// the block is decoded, whether it is a source symbol).
    pub fn has_symbol(&self, esi: u32) -> bool {
        if esi < self.source_symbols {
            self.decoded || self.received_source[esi as usize / 64] & (1 << (esi % 64)) != 0
        } else {
            self.repair_esis.contains(&esi)
        }
    }

    /// Add the received symbol with encoding symbol ID `esi` to the current
    /// block, copying `data` into the decoder.
    ///
    /// Returns `Ok(true)` if the symbol was new, and `Ok(false)` if it was
    /// already added or the block is already decoded; in both of those cases
    /// nothing changes. Adding never decodes: call
    /// [`try_decode`](Self::try_decode) after a batch of symbols.
    ///
    /// # Errors
    /// [`BlockError::NoBlock`] before the first reset,
    /// [`BlockError::SymbolSizeMismatch`] unless `data.len() == T`, and
    /// [`BlockError::InvalidEsi`] if `esi` is not below 2^24. The decoder is
    /// then unchanged.
    pub fn add_symbol(&mut self, esi: u32, data: &[u8]) -> Result<bool, BlockError> {
        if self.source_symbols == 0 {
            return Err(BlockError::NoBlock);
        }
        let symbol_size = self.layout.symbol_size();
        if data.len() != symbol_size {
            return Err(BlockError::SymbolSizeMismatch {
                len: data.len(),
                expected: symbol_size,
            });
        }
        if esi >= ESI_LIMIT {
            return Err(BlockError::InvalidEsi(esi));
        }
        if self.decoded || self.has_symbol(esi) {
            return Ok(false);
        }
        if esi < self.source_symbols {
            self.source.get_mut(esi as usize).copy_from_slice(data);
            self.received_source[esi as usize / 64] |= 1 << (esi % 64);
            self.received_source_count += 1;
        } else {
            let row = self.repair_esis.len();
            self.repair_esis.push(esi);
            self.repair.resize_symbols(row + 1);
            self.repair.get_mut(row).copy_from_slice(data);
        }
        self.unattempted = true;
        Ok(true)
    }

    /// Try to recover the current block from the symbols added so far.
    ///
    /// Returns `Ok(true)` once the block is decoded (immediately, without
    /// work, if it already was), and `Ok(false)` if the symbols added so far
    /// are not enough. The decoder only works when a symbol was added since
    /// its last attempt and at least K distinct symbols are present; with all
    /// K source symbols present it decodes without solving.
    ///
    /// # Errors
    /// [`BlockError::NoBlock`] before the first reset.
    pub fn try_decode(&mut self) -> Result<bool, BlockError> {
        if self.source_symbols == 0 {
            return Err(BlockError::NoBlock);
        }
        if self.decoded {
            return Ok(true);
        }
        if !self.unattempted || self.received_symbols() < self.source_symbols {
            return Ok(false);
        }
        self.unattempted = false;
        if self.received_source_count < self.source_symbols {
            self.solves += 1;
            if !self.solve() {
                return Ok(false);
            }
        }
        self.decoded = true;
        Ok(true)
    }

    /// Copy bytes `[offset, offset + out.len())` of the decoded source block
    /// into `out`. The block is `K * T` bytes in block order (RFC 6330
    /// section 4.4.1.2): with one sub-block, source symbol `i` is bytes
    /// `[i * T, (i + 1) * T)`, so `offset = first * T` and
    /// `out.len() = count * T` copy symbols `first..first + count`.
    ///
    /// # Errors
    /// [`BlockError::NoBlock`], [`BlockError::NotDecoded`], and
    /// [`BlockError::OutOfRange`] if the range ends past `K * T`. On error
    /// nothing is written.
    pub fn copy_block_into(&self, offset: usize, out: &mut [u8]) -> Result<(), BlockError> {
        if self.source_symbols == 0 {
            return Err(BlockError::NoBlock);
        }
        if !self.decoded {
            return Err(BlockError::NotDecoded);
        }
        let k = self.source_symbols as usize;
        let block_len = k * self.layout.symbol_size();
        let end = offset
            .checked_add(out.len())
            .filter(|&end| end <= block_len);
        let Some(end) = end else {
            return Err(BlockError::OutOfRange {
                offset,
                len: out.len(),
                block_len,
            });
        };
        let symbols = self.source.as_bytes();
        self.layout
            .for_each_run(k, offset, end, |block_offset, symbol_offset, len| {
                let dest = block_offset - offset;
                out[dest..dest + len].copy_from_slice(&symbols[symbol_offset..symbol_offset + len]);
            });
        Ok(())
    }

    /// Solve for the intermediate symbols and rebuild the missing source
    /// symbols into their slots. Mirrors `SourceBlockDecoder::decode` cases
    /// 3a (GF(2) only, when the overhead allows) and 3b (with HDPC rows).
    #[allow(non_snake_case)]
    fn solve(&mut self) -> bool {
        let k = self.source_symbols;
        let symbol_size = self.layout.symbol_size();
        let k_ext = extended_source_block_symbols(k);
        let padding = (k_ext - k) as usize;
        let S = num_ldpc_symbols(k) as usize;
        let H = num_hdpc_symbols(k) as usize;
        let L = num_intermediate_symbols(k) as usize;

        let mut isis = Vec::with_capacity(
            self.received_source_count as usize + padding + self.repair_esis.len(),
        );
        for i in 0..k {
            if self.received_source[i as usize / 64] & (1 << (i % 64)) != 0 {
                isis.push(i);
            }
        }
        isis.extend(k..k_ext);
        isis.extend(self.repair_esis.iter().map(|&esi| esi + (k_ext - k)));

        let fill = |first_row: usize| -> SymbolSlab {
            let mut d = SymbolSlab::with_zeros(first_row + isis.len(), symbol_size);
            let mut row = first_row;
            for i in 0..k as usize {
                if self.received_source[i / 64] & (1 << (i % 64)) != 0 {
                    d.get_mut(row).copy_from_slice(self.source.get(i));
                    row += 1;
                }
            }
            row += padding;
            for r in 0..self.repair_esis.len() {
                d.get_mut(row).copy_from_slice(self.repair.get(r));
                row += 1;
            }
            d
        };

        let mut intermediate = None;
        if S + isis.len() >= L {
            let d = fill(S);
            intermediate = if k_ext >= SPARSE_MATRIX_THRESHOLD {
                let a = generate_constraint_matrix_no_hdpc::<SparseBinaryMatrix>(k, &isis);
                fused_inverse_mul_symbols_no_hdpc(a, d, k).0
            } else {
                let a = generate_constraint_matrix_no_hdpc::<DenseBinaryMatrix>(k, &isis);
                fused_inverse_mul_symbols_no_hdpc(a, d, k).0
            };
        }
        if intermediate.is_none() {
            let d = fill(S + H);
            intermediate = if k_ext >= SPARSE_MATRIX_THRESHOLD {
                let (a, hdpc) = generate_constraint_matrix::<SparseBinaryMatrix>(k, &isis);
                fused_inverse_mul_symbols(a, hdpc, d, k).0
            } else {
                let (a, hdpc) = generate_constraint_matrix::<DenseBinaryMatrix>(k, &isis);
                fused_inverse_mul_symbols(a, hdpc, d, k).0
            };
        }
        let Some(intermediate) = intermediate else {
            return false;
        };

        let lt_symbols = num_lt_symbols(k);
        let pi_symbols = num_pi_symbols(k);
        let sys_index = systematic_index(k);
        let p1 = calculate_p1(k);
        for i in 0..k {
            if self.received_source[i as usize / 64] & (1 << (i % 64)) != 0 {
                continue;
            }
            let dest = self.source.get_mut(i as usize);
            let tuple = intermediate_tuple(i, lt_symbols, sys_index, p1);
            let mut first = true;
            enc_indices(tuple, lt_symbols, pi_symbols, p1, |j| {
                if first {
                    dest.copy_from_slice(intermediate.get(j));
                    first = false;
                } else {
                    crate::octets::add_assign(dest, intermediate.get(j));
                }
            });
        }
        true
    }
}
