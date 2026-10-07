#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "std")]
use std::{sync::Arc, vec::Vec};

use crate::base::ObjectTransmissionInformation;
use crate::base::intermediate_tuple;
use crate::encoder::{SourceBlockEncodingPlan, enc_into};
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
/// Every block is encoded by replaying a [`SourceBlockEncodingPlan`] for its
/// extended size K' (RFC 6330 section 5.3.1) into the encoder's own
/// storage. A plan depends only on K', so it serves every K with that K'.
/// The plan comes from, in order:
///
/// 1. the attached plan ([`with_plan`](Self::with_plan) or
///    [`set_plan`](Self::set_plan)), typically the full block's from
///    [`SourceBlockEncodingPlan::cached`];
/// 2. the encoder's plan memo: plans it generated itself for earlier
///    blocks, such as the short final blocks of a stream;
/// 3. otherwise, the encoder solves the block's constraint matrix once to
///    generate the plan (this allocates) and keeps it in the memo if it fits
///    the memo's byte budget ([`set_plan_memo_limit`](Self::set_plan_memo_limit),
///    [`DEFAULT_PLAN_MEMO_BYTES`](Self::DEFAULT_PLAN_MEMO_BYTES) by default).
///    A plan that does not fit is used once and dropped; the memo never
///    evicts.
///
/// The memo is private to the encoder: the process-wide plan cache is never
/// read or written. Its size is bounded by the number of distinct K' values
/// RFC 6330 defines up to the largest block encoded. A stream of 128-symbol
/// blocks (K' = 138) has short final blocks of 27 distinct K' (10 to 127),
/// whose plans take about 1.7 MB together.
///
/// So once the storage has grown to the largest block it sees and each K'
/// it sees has a plan, encoding a block and generating symbols do not
/// allocate.
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
    /// K' of the attached plan; 0 when none is attached.
    plan_extended: u32,
    /// Plans this encoder generated, as `(K', plan)` sorted by K'.
    memo: Vec<(u32, Arc<SourceBlockEncodingPlan>)>,
    /// The heap bytes of the plans in `memo`.
    memo_bytes: usize,
    /// The byte budget of `memo`.
    memo_limit: usize,
}

impl ReusableSourceBlockEncoder {
    /// The default byte budget of the plan memo, 4 MiB: enough for a plan
    /// for every K' up to 217 (4,102,832 bytes), which covers every short
    /// block of a stream of blocks of up to 218 symbols.
    pub const DEFAULT_PLAN_MEMO_BYTES: usize = 4 << 20;

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
            plan_extended: 0,
            memo: Vec::new(),
            memo_bytes: 0,
            memo_limit: Self::DEFAULT_PLAN_MEMO_BYTES,
            layout,
        })
    }

    /// Create an encoder with no block loaded that encodes blocks with the
    /// extended size K' of `plan.source_symbol_count()` with `plan`. See
    /// [`new`](Self::new) for `config` and the errors.
    pub fn with_plan(
        config: &ObjectTransmissionInformation,
        plan: Arc<SourceBlockEncodingPlan>,
    ) -> Result<Self, BlockError> {
        let mut encoder = Self::new(config)?;
        encoder.set_plan(Some(plan));
        Ok(encoder)
    }

    /// Attach `plan`, replacing any plan already attached, or detach it with
    /// `None`. This does not change the loaded block.
    pub fn set_plan(&mut self, plan: Option<Arc<SourceBlockEncodingPlan>>) {
        self.plan_extended = plan.as_ref().map_or(0, |plan| {
            extended_source_block_symbols(u32::from(plan.source_symbol_count()))
        });
        self.plan = plan;
    }

    /// The attached plan, if any.
    pub fn plan(&self) -> Option<&Arc<SourceBlockEncodingPlan>> {
        self.plan.as_ref()
    }

    /// Set the byte budget of the plan memo (see the type documentation).
    /// Plans already in the memo are dropped, largest K' first, until the
    /// memo fits; 0 empties the memo and stops it from keeping plans.
    pub fn set_plan_memo_limit(&mut self, bytes: usize) {
        self.memo_limit = bytes;
        while self.memo_bytes > bytes {
            match self.memo.pop() {
                Some((_, plan)) => self.memo_bytes -= plan.heap_bytes(),
                None => break,
            }
        }
    }

    /// The byte budget of the plan memo.
    pub fn plan_memo_limit(&self) -> usize {
        self.memo_limit
    }

    /// The number of plans in the memo.
    pub fn plan_memo_len(&self) -> usize {
        self.memo.len()
    }

    /// The heap bytes held by the plans in the memo.
    pub fn plan_memo_bytes(&self) -> usize {
        self.memo_bytes
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

    /// Prepare for a block of `source_symbols` symbols now, so that
    /// encoding the first block of that size does not allocate: grow the
    /// storage and, if no plan covers its K', generate one into the memo
    /// (when it fits the memo's budget).
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
        let extended = extended_source_block_symbols(source_symbols);
        if !self.has_plan_for(extended) {
            let plan = Self::generate_plan(source_symbols);
            self.remember(extended, plan);
        }
        Ok(())
    }

    /// Source symbol `index` of the loaded block, borrowed from the
    /// encoder: the same bytes [`repair_into`](Self::repair_into) writes for
    /// ESI `index`. `None` when no block is loaded or `index >= K`.
    pub fn source_symbol(&self, index: u32) -> Option<&[u8]> {
        (index < self.source_symbols).then(|| self.source.get(index as usize))
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
    /// The block is encoded with the plan for its K' (see the type
    /// documentation). Every plan gives the same symbols as
    /// `SourceBlockEncoder`.
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

        let extended = extended_source_block_symbols(source_symbols);
        let mut transient = None;
        if !self.has_plan_for(extended) {
            // The only allocating step: generate the plan for this K' (one
            // solve) and keep it if the memo has room.
            let plan = Self::generate_plan(source_symbols);
            if !self.remember(extended, Arc::clone(&plan)) {
                transient = Some(plan);
            }
        }
        let plan = match transient.as_deref() {
            Some(plan) => plan,
            None => find_plan(&self.plan, self.plan_extended, &self.memo, extended)
                .expect("a plan covers this K'"),
        };
        load_intermediate(
            plan.operations(),
            source_symbols,
            &self.source,
            &mut self.intermediate,
            &mut self.spare_order,
        );
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

    fn has_plan_for(&self, extended: u32) -> bool {
        find_plan(&self.plan, self.plan_extended, &self.memo, extended).is_some()
    }

    fn generate_plan(source_symbols: u32) -> Arc<SourceBlockEncodingPlan> {
        // K <= MAX_SOURCE_SYMBOLS_PER_BLOCK (56403) fits a u16.
        let mut plan = SourceBlockEncodingPlan::generate(source_symbols as u16);
        plan.shrink_to_fit();
        Arc::new(plan)
    }

    /// Keep `plan` for blocks of size `extended` if it fits the memo's
    /// budget. Returns whether it was kept.
    fn remember(&mut self, extended: u32, plan: Arc<SourceBlockEncodingPlan>) -> bool {
        let bytes = plan.heap_bytes();
        if self.memo_bytes.saturating_add(bytes) > self.memo_limit {
            return false;
        }
        match self.memo.binary_search_by_key(&extended, |&(k, _)| k) {
            Ok(_) => false,
            Err(i) => {
                self.memo.insert(i, (extended, plan));
                self.memo_bytes += bytes;
                true
            }
        }
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

/// The plan for blocks with extended size `extended`: the attached one if it
/// covers that K', else the memo's.
fn find_plan<'a>(
    attached: &'a Option<Arc<SourceBlockEncodingPlan>>,
    attached_extended: u32,
    memo: &'a [(u32, Arc<SourceBlockEncodingPlan>)],
    extended: u32,
) -> Option<&'a SourceBlockEncodingPlan> {
    match attached {
        Some(plan) if attached_extended == extended => Some(plan),
        _ => memo
            .binary_search_by_key(&extended, |&(k, _)| k)
            .ok()
            .map(|i| &*memo[i].1),
    }
}

/// Compute the L intermediate symbols of a K-symbol block into
/// `intermediate` by replaying `operations`, a plan for the block's K'
/// (section 5.3.3.4): D is S + H zero symbols, the K source symbols, then
/// zero padding up to K'. Reuses the storage of `intermediate` and
/// `spare_order`; allocates only when they must grow.
fn load_intermediate(
    operations: &[SymbolOps],
    source_symbols: u32,
    source: &SymbolSlab,
    intermediate: &mut SymbolSlab,
    spare_order: &mut Vec<usize>,
) {
    let l = num_intermediate_symbols(source_symbols) as usize;
    let s = num_ldpc_symbols(source_symbols) as usize;
    let h = num_hdpc_symbols(source_symbols) as usize;
    if let Some(order) = intermediate.reset_zeroed(l, source.symbol_size()) {
        *spare_order = order;
    }
    intermediate.copy_block_from(s + h, source.as_bytes());
    for op in operations {
        match op {
            SymbolOps::Reorder { order } => {
                let mut mapping = core::mem::take(spare_order);
                mapping.clear();
                mapping.extend_from_slice(order);
                if let Some(previous) = intermediate.replace_reorder(mapping) {
                    *spare_order = previous;
                }
            }
            op => perform_op(op, intermediate),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::encoder::SourceBlockEncodingPlan;
    use crate::systematic_constants::{
        SYSTEMATIC_INDICES_AND_PARAMETERS, extended_source_block_symbols,
    };

    /// The reason one plan serves every K with the same K': the operations
    /// are identical. Checked for every K up to 400 (the dense and sparse
    /// solvers) in release builds; debug builds, where plan generation is
    /// slow, check K up to 60 and the first sparse K' (257).
    #[test]
    fn plan_operations_depend_only_on_the_extended_size() {
        let ks: &[core::ops::RangeInclusive<u32>] = if cfg!(debug_assertions) {
            &[1..=60, 250..=257]
        } else {
            &[1..=400]
        };
        let mut previous: Option<(u32, SourceBlockEncodingPlan)> = None;
        for k in ks.iter().cloned().flatten() {
            let extended = extended_source_block_symbols(k);
            let plan = SourceBlockEncodingPlan::generate(k as u16);
            if let Some((prev_extended, prev)) = &previous
                && *prev_extended == extended
            {
                assert_eq!(plan.operations(), prev.operations(), "k={k}");
                continue;
            }
            assert!(
                SYSTEMATIC_INDICES_AND_PARAMETERS
                    .iter()
                    .any(|&(kp, ..)| kp == extended)
            );
            previous = Some((extended, plan));
        }
    }

    #[test]
    fn shrunk_plan_reports_its_heap_bytes() {
        let mut plan = SourceBlockEncodingPlan::generate(128);
        let before = plan.heap_bytes();
        plan.shrink_to_fit();
        let after = plan.heap_bytes();
        assert!(after < before, "{after} < {before}");
        assert!(after >= core::mem::size_of_val(plan.operations()));
    }
}
