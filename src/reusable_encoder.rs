#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::{sync::Arc, vec::Vec};

use crate::base::ObjectTransmissionInformation;
use crate::base::intermediate_tuple;
use crate::encoder::{
    SPARSE_MATRIX_THRESHOLD, SourceBlockEncodingPlan, enc_into, gen_intermediate_symbols,
};
use crate::operation_vector::{SymbolOps, perform_op};
use crate::reusable::{BlockError, ESI_LIMIT, SubBlockLayout, check_source_symbols, strided_len};
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_intermediate_symbols,
    num_ldpc_symbols, num_lt_symbols, systematic_index,
};

/// An RFC 6330 source block encoder that is loaded with one source block
/// after another and keeps its storage between them.
///
/// It produces the same symbols as [`SourceBlockEncoder`](crate::SourceBlockEncoder)
/// for the same block, but it writes them into caller buffers instead of
/// returning [`EncodingPacket`](crate::EncodingPacket)s, and
/// [`encode_block`](Self::encode_block) reuses the buffers of the previous
/// block. Once its storage has grown to the largest block it sees, encoding
/// a block with the attached plan and generating symbols do not allocate.
///
/// The encoder holds at most one [`SourceBlockEncodingPlan`], for the block
/// size it encodes most often (typically the full block of a stream). A
/// block of that size is encoded with the plan; any other size is solved
/// directly, without generating or caching a plan for it.
///
/// ```
/// use raptorq::{ObjectTransmissionInformation, ReusableSourceBlockEncoder};
///
/// let config = ObjectTransmissionInformation::new(0, 8, 1, 1, 8);
/// let mut encoder = ReusableSourceBlockEncoder::new(&config).unwrap();
/// let data = [7u8; 20]; // 3 symbols, the last one zero-extended
/// encoder.encode_block(&data, 3).unwrap();
/// let mut repair = [0u8; 2 * 8];
/// encoder.repair_range_into(3, 2, &mut repair, 8).unwrap(); // ESIs 3 and 4
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReusableSourceBlockEncoder {
    layout: SubBlockLayout,
    /// K of the loaded block; 0 when no block is loaded.
    source_symbols: u32,
    /// The loaded block's K source symbols.
    source: SymbolSlab,
    /// The loaded block's L intermediate symbols.
    intermediate: SymbolSlab,
    /// Storage for the intermediate symbols' reorder mapping, kept between
    /// blocks.
    spare_order: Vec<usize>,
    plan: Option<Arc<SourceBlockEncodingPlan>>,
}

impl ReusableSourceBlockEncoder {
    /// Create an encoder with no block loaded.
    ///
    /// Only the symbol size, sub-block count and symbol alignment of `config`
    /// are used; its transfer length and source block count are ignored,
    /// because each block's size is given to
    /// [`encode_block`](Self::encode_block).
    ///
    /// # Errors
    /// [`BlockError::InvalidConfig`] if the symbol size, alignment or
    /// sub-block count is zero, or the symbol size is not a multiple of the
    /// alignment.
    pub fn new(config: &ObjectTransmissionInformation) -> Result<Self, BlockError> {
        let layout = SubBlockLayout::new(config)?;
        Ok(ReusableSourceBlockEncoder {
            source_symbols: 0,
            source: SymbolSlab::with_zeros(0, layout.symbol_size()),
            intermediate: SymbolSlab::with_zeros(0, layout.symbol_size()),
            spare_order: Vec::new(),
            plan: None,
            layout,
        })
    }

    /// Create an encoder with no block loaded that encodes blocks of
    /// `plan.source_symbol_count()` symbols with `plan`. See
    /// [`new`](Self::new) for `config` and the errors.
    pub fn with_plan(
        config: &ObjectTransmissionInformation,
        plan: Arc<SourceBlockEncodingPlan>,
    ) -> Result<Self, BlockError> {
        let mut encoder = Self::new(config)?;
        encoder.plan = Some(plan);
        Ok(encoder)
    }

    /// Attach `plan`, replacing any plan already attached, or detach it with
    /// `None`. This does not change the loaded block.
    pub fn set_plan(&mut self, plan: Option<Arc<SourceBlockEncodingPlan>>) {
        self.plan = plan;
    }

    /// The attached plan, if any.
    pub fn plan(&self) -> Option<&Arc<SourceBlockEncodingPlan>> {
        self.plan.as_ref()
    }

    /// The symbol size T in bytes.
    pub fn symbol_size(&self) -> u16 {
        self.layout.symbol_size() as u16
    }

    /// The number of source symbols K of the loaded block, or 0 when no
    /// block is loaded.
    pub fn source_symbols(&self) -> u32 {
        self.source_symbols
    }

    /// Unload the block, keeping the storage. Symbol generation then fails
    /// with [`BlockError::NoBlock`] until the next
    /// [`encode_block`](Self::encode_block).
    pub fn clear(&mut self) {
        self.source_symbols = 0;
    }

    /// Grow the storage to hold a block of `source_symbols` symbols now, so
    /// that encoding the first block of that size does not allocate.
    ///
    /// # Errors
    /// [`BlockError::InvalidSourceSymbols`] if `source_symbols` is zero or
    /// above `MAX_SOURCE_SYMBOLS_PER_BLOCK`.
    pub fn reserve(&mut self, source_symbols: u32) -> Result<(), BlockError> {
        check_source_symbols(source_symbols)?;
        let l = num_intermediate_symbols(source_symbols) as usize;
        self.source.reserve_symbols(source_symbols as usize);
        self.intermediate.reserve_symbols(l);
        if self.spare_order.capacity() < l {
            self.spare_order.reserve(l - self.spare_order.len());
        }
        Ok(())
    }

    /// Load a new source block of `source_symbols` (K) symbols and compute
    /// its intermediate symbols, replacing the loaded block.
    ///
    /// `data` is the block's bytes, at most `K * T` long. Bytes past the end
    /// of `data` are zero: a short `data` gives a zero-extended last symbol
    /// and, if shorter still, trailing all-zero symbols, exactly as if
    /// `data` had been zero-padded to `K * T` bytes and passed to
    /// [`SourceBlockEncoder::new`](crate::SourceBlockEncoder::new). With
    /// more than one sub-block, `data` is in block order (RFC 6330 section
    /// 4.4.1.2), as for `SourceBlockEncoder`.
    ///
    /// If a plan is attached and `plan.source_symbol_count() == K`, the plan
    /// is used; otherwise the block is solved directly. Both give the same
    /// symbols.
    ///
    /// # Errors
    /// - [`BlockError::InvalidSourceSymbols`] if K is zero or above
    ///   `MAX_SOURCE_SYMBOLS_PER_BLOCK`;
    /// - [`BlockError::DataTooLong`] if `data.len() > K * T`.
    ///
    /// On error the encoder is unchanged: the previously loaded block, if
    /// any, stays loaded.
    pub fn encode_block(&mut self, data: &[u8], source_symbols: u32) -> Result<(), BlockError> {
        check_source_symbols(source_symbols)?;
        let symbol_size = self.layout.symbol_size();
        let k = source_symbols as usize;
        let block_len = k * symbol_size;
        if data.len() > block_len {
            return Err(BlockError::DataTooLong {
                len: data.len(),
                max: block_len,
            });
        }

        // Source symbols, zero-extended and sub-block packed.
        self.source.reset_zeroed(k, symbol_size);
        {
            let symbols = self.source.as_bytes_mut();
            self.layout
                .for_each_run(k, 0, data.len(), |block_offset, symbol_offset, len| {
                    symbols[symbol_offset..symbol_offset + len]
                        .copy_from_slice(&data[block_offset..block_offset + len]);
                });
        }

        let plan = self
            .plan
            .as_ref()
            .filter(|plan| u32::from(plan.source_symbol_count()) == source_symbols);
        if let Some(plan) = plan {
            // D: S + H zero rows, the K source symbols, then zero padding up to
            // K' (section 5.3.3.4), transformed in place by the plan.
            let l = num_intermediate_symbols(source_symbols) as usize;
            let s = num_ldpc_symbols(source_symbols) as usize;
            let h = num_hdpc_symbols(source_symbols) as usize;
            if let Some(order) = self.intermediate.reset_zeroed(l, symbol_size) {
                self.spare_order = order;
            }
            self.intermediate
                .copy_block_from(s + h, self.source.as_bytes());
            for op in plan.operations() {
                match op {
                    SymbolOps::Reorder { order } => {
                        let mut mapping = core::mem::take(&mut self.spare_order);
                        mapping.clear();
                        mapping.extend_from_slice(order);
                        if let Some(previous) = self.intermediate.replace_reorder(mapping) {
                            self.spare_order = previous;
                        }
                    }
                    op => perform_op(op, &mut self.intermediate),
                }
            }
        } else {
            let (intermediate, _) =
                gen_intermediate_symbols(&self.source, symbol_size, SPARSE_MATRIX_THRESHOLD);
            self.intermediate = intermediate
                .expect("the RFC 6330 constraint matrix of a source block is always invertible");
        }
        self.source_symbols = source_symbols;
        Ok(())
    }

    /// Write the encoding symbols with the given ESIs into `out`, symbol `i`
    /// at `out[i * stride..i * stride + T]`. Bytes between symbols are left
    /// untouched, so a caller can reserve per-symbol headers by passing a
    /// sub-slice of its buffer and a stride of header + T.
    ///
    /// An ESI below K gives that source symbol; an ESI of K or more gives
    /// repair symbol `esi - K` (RFC 6330 section 5.3.2: internal symbol ID
    /// `esi + K' - K`). ESIs may repeat and come in any order.
    ///
    /// `out` must be at least `(esis.len() - 1) * stride + T` bytes (0 when
    /// `esis` is empty).
    ///
    /// # Errors
    /// [`BlockError::NoBlock`], [`BlockError::InvalidStride`] if
    /// `stride < T`, [`BlockError::InvalidEsi`] for the first ESI not below
    /// 2^24, and [`BlockError::BufferTooSmall`]. On error nothing is written.
    pub fn repair_into(
        &self,
        esis: &[u32],
        out: &mut [u8],
        stride: usize,
    ) -> Result<(), BlockError> {
        self.check_output(esis.len(), out.len(), stride)?;
        if let Some(&esi) = esis.iter().find(|&&esi| esi >= ESI_LIMIT) {
            return Err(BlockError::InvalidEsi(esi));
        }
        let symbol_size = self.layout.symbol_size();
        for (i, &esi) in esis.iter().enumerate() {
            let start = i * stride;
            self.symbol_into(esi, &mut out[start..start + symbol_size]);
        }
        Ok(())
    }

    /// Write the `count` encoding symbols with ESIs `first_esi`,
    /// `first_esi + 1`, ... into `out`, laid out as in
    /// [`repair_into`](Self::repair_into). To generate repair symbols
    /// `r..r + n` of the block, pass `first_esi = K + r`.
    ///
    /// # Errors
    /// As [`repair_into`](Self::repair_into); [`BlockError::InvalidEsi`]
    /// reports the first ESI of the range that is not below 2^24. On error
    /// nothing is written.
    pub fn repair_range_into(
        &self,
        first_esi: u32,
        count: u32,
        out: &mut [u8],
        stride: usize,
    ) -> Result<(), BlockError> {
        self.check_output(count as usize, out.len(), stride)?;
        if count > 0 {
            let last = u64::from(first_esi) + u64::from(count) - 1;
            if last >= u64::from(ESI_LIMIT) {
                return Err(BlockError::InvalidEsi(first_esi.max(ESI_LIMIT)));
            }
        }
        let symbol_size = self.layout.symbol_size();
        for i in 0..count {
            let start = i as usize * stride;
            self.symbol_into(first_esi + i, &mut out[start..start + symbol_size]);
        }
        Ok(())
    }

    fn check_output(&self, count: usize, out_len: usize, stride: usize) -> Result<(), BlockError> {
        if self.source_symbols == 0 {
            return Err(BlockError::NoBlock);
        }
        let needed = strided_len(count, self.layout.symbol_size(), stride)?;
        if out_len < needed {
            return Err(BlockError::BufferTooSmall {
                len: out_len,
                needed,
            });
        }
        Ok(())
    }

    fn symbol_into(&self, esi: u32, dest: &mut [u8]) {
        let k = self.source_symbols;
        if esi < k {
            dest.copy_from_slice(self.source.get(esi as usize));
            return;
        }
        let isi = esi + (extended_source_block_symbols(k) - k);
        let tuple =
            intermediate_tuple(isi, num_lt_symbols(k), systematic_index(k), calculate_p1(k));
        enc_into(dest, k, &self.intermediate, tuple);
    }
}
