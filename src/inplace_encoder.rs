#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
use core::mem::MaybeUninit;
use core::ptr;
#[cfg(feature = "std")]
use std::sync::{Mutex, OnceLock};
#[cfg(feature = "std")]
use std::{sync::Arc, vec::Vec};

use crate::base::{ObjectTransmissionInformation, intermediate_tuple};
use crate::encoder::{SourceBlockEncodingPlan, enc_into_uninit};
use crate::operation_vector::SymbolOps;
use crate::reusable::{BlockError, ESI_LIMIT, SubBlockLayout, check_source_symbols, strided_len};
use crate::symbol_slab::SymbolSlab;
use crate::systematic_constants::{
    calculate_p1, extended_source_block_symbols, num_hdpc_symbols, num_intermediate_symbols,
    num_ldpc_symbols, num_lt_symbols, systematic_index,
};

/// The encoding plans an [`InPlaceBlockEncoder`] generates for blocks closed
/// short of the full-block K' (RFC 6330 section 5.3.1), kept for reuse.
///
/// One memo is meant to be shared by every encoder of a stream: a plan
/// depends only on K', so it serves every block with that K'. A plan is
/// generated (one solve, which allocates) the first time a K' is closed and
/// kept if it fits the memo's byte budget; a plan that does not fit is used
/// for that block and dropped. The memo never evicts.
///
/// Memos only affect speed and allocation, never the symbols, so any two
/// are equal (`==`).
#[derive(Clone, Debug)]
pub struct EncodingPlanMemo {
    /// `(K', plan)`, sorted by K'.
    plans: Vec<(u32, Arc<SourceBlockEncodingPlan>)>,
    /// The heap bytes of the plans in `plans`.
    bytes: usize,
    /// The byte budget of `plans`.
    limit: usize,
}

impl PartialEq for EncodingPlanMemo {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for EncodingPlanMemo {}

impl EncodingPlanMemo {
    /// An empty memo that keeps plans up to `limit` heap bytes in total.
    pub fn new(limit: usize) -> Self {
        EncodingPlanMemo {
            plans: Vec::new(),
            bytes: 0,
            limit,
        }
    }

    /// The memo's byte budget.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// The number of plans kept.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// Whether no plan is kept.
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    /// The heap bytes held by the kept plans.
    pub fn heap_bytes(&self) -> usize {
        self.bytes
    }

    /// Generate and keep the plan for blocks of `source_symbols` symbols now,
    /// so the first close of that K' does not solve. Returns whether a plan
    /// for that K' is kept.
    ///
    /// # Errors
    /// [`BlockError::InvalidSourceSymbols`] if `source_symbols` is zero or
    /// above `MAX_SOURCE_SYMBOLS_PER_BLOCK`.
    pub fn reserve(&mut self, source_symbols: u32) -> Result<bool, BlockError> {
        check_source_symbols(source_symbols)?;
        let extended = extended_source_block_symbols(source_symbols);
        if self.find(extended).is_none() {
            let plan = generate_plan(source_symbols);
            self.remember(extended, plan);
        }
        Ok(self.find(extended).is_some())
    }

    /// The plan for blocks of `source_symbols` symbols: the kept one, or a
    /// new one, kept if it fits.
    pub(crate) fn plan_for(&mut self, source_symbols: u32) -> Arc<SourceBlockEncodingPlan> {
        let extended = extended_source_block_symbols(source_symbols);
        if let Some(plan) = self.find(extended) {
            return Arc::clone(plan);
        }
        let plan = generate_plan(source_symbols);
        self.remember(extended, Arc::clone(&plan));
        plan
    }

    fn find(&self, extended: u32) -> Option<&Arc<SourceBlockEncodingPlan>> {
        self.plans
            .binary_search_by_key(&extended, |&(k, _)| k)
            .ok()
            .map(|i| &self.plans[i].1)
    }

    fn remember(&mut self, extended: u32, plan: Arc<SourceBlockEncodingPlan>) {
        let bytes = plan.heap_bytes();
        if self.bytes.saturating_add(bytes) > self.limit {
            return;
        }
        if let Err(i) = self.plans.binary_search_by_key(&extended, |&(k, _)| k) {
            self.plans.insert(i, (extended, plan));
            self.bytes += bytes;
        }
    }
}

/// The largest K' the process-wide plan store keeps: K'(256).
pub(crate) const STORE_MAX_EXTENDED_SYMBOLS: u32 = 257;

/// The byte budget of the process-wide plan store: the 46 plans for
/// K' <= 257, shrunk, measured on 64-bit targets.
pub(crate) const STORE_BYTES: usize = 5_291_880;

#[cfg(feature = "std")]
fn plan_store() -> &'static Mutex<EncodingPlanMemo> {
    static STORE: OnceLock<Mutex<EncodingPlanMemo>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(EncodingPlanMemo::new(STORE_BYTES)))
}

/// The process-wide plan for blocks of `source_symbols` symbols, keyed by
/// K', generated on the first request for each K'. `None` above
/// [`STORE_MAX_EXTENDED_SYMBOLS`]. The lock is held only to look up and to
/// insert, never while a plan is generated; if two threads generate the same
/// K', the first insert is kept.
///
/// Separate from [`SourceBlockEncodingPlan::cached`], which is keyed by K and
/// evicts: whole-object plans there would evict the full-block plans its
/// callers fetch.
#[cfg(feature = "std")]
pub(crate) fn stored_plan(source_symbols: u32) -> Option<Arc<SourceBlockEncodingPlan>> {
    let extended = extended_source_block_symbols(source_symbols);
    if extended > STORE_MAX_EXTENDED_SYMBOLS {
        return None;
    }
    let lock = || {
        plan_store()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
    if let Some(plan) = lock().find(extended) {
        return Some(Arc::clone(plan));
    }
    let plan = generate_plan(extended);
    let mut store = lock();
    if let Some(kept) = store.find(extended) {
        return Some(Arc::clone(kept));
    }
    store.remember(extended, Arc::clone(&plan));
    Some(plan)
}

/// The heap bytes and plan count of the process-wide plan store.
#[cfg(all(feature = "std", test))]
pub(crate) fn plan_store_usage() -> (usize, usize) {
    let store = plan_store().lock().unwrap();
    (store.heap_bytes(), store.len())
}

fn generate_plan(source_symbols: u32) -> Arc<SourceBlockEncodingPlan> {
    // K <= MAX_SOURCE_SYMBOLS_PER_BLOCK (56403) fits a u16.
    let mut plan = SourceBlockEncodingPlan::generate(source_symbols as u16);
    plan.shrink_to_fit();
    Arc::new(plan)
}

/// A closed block: its source symbol count k, the first physical row of its
/// L(k)-row window, and the plan whose operations were replayed over it.
#[derive(Clone, Debug)]
struct Closed {
    source_symbols: u32,
    base: usize,
    plan: Arc<SourceBlockEncodingPlan>,
}

/// An RFC 6330 source block encoder that builds the block's D vector
/// (section 5.3.3.4) in place: the caller writes each source symbol straight
/// into its row of D, and closing the block solves for the intermediate
/// symbols C over those same rows. Nothing copies a source symbol a second
/// time.
///
/// The encoder holds one slab of L(K) symbols, where K is the largest block
/// it takes. Source symbol `i` lives at row `S + H + i`, with S and H those
/// of K. A block closed at k <= K source symbols uses the L(k)-row window
/// that starts at row `(S + H)(K) - (S + H)(k)`: its S + H rows sit just
/// before the source rows, and its K'(k) - k padding rows just after. Since
/// S + H never decreases as K' grows, the window always fits the slab.
///
/// Closing zeroes only the window's L(k) - k non-source rows and replays a
/// [`SourceBlockEncodingPlan`] for K'(k) over the window: the full-block plan
/// given at construction when K'(k) = K'(K), otherwise the plan from an
/// [`EncodingPlanMemo`]. The plan's final reorder is not applied; its
/// mapping is borrowed from the plan to address C, so a close copies no
/// index data. Once the memo holds the plan for each K' seen, admitting,
/// closing and generating symbols do not allocate.
///
/// A closed block keeps its rows until [`reopen`](Self::reopen). Its source
/// symbols are then read back by encoding (systematic: section 5.3.3.4.1),
/// not from the overwritten rows.
///
/// Two encoders are equal (`==`) when they have the same K and T and the
/// same closed block, or neither has one closed and their slabs are equal.
#[derive(Clone, Debug)]
pub struct InPlaceBlockEncoder {
    symbol_size: usize,
    /// K: the most source symbols a block may hold.
    max_source_symbols: u32,
    /// S + H of K: the row of source symbol 0.
    offset: usize,
    /// L(K) rows. Never has a reorder mapping.
    slab: SymbolSlab,
    full_plan: Arc<SourceBlockEncodingPlan>,
    closed: Option<Closed>,
}

impl PartialEq for InPlaceBlockEncoder {
    fn eq(&self, other: &Self) -> bool {
        self.symbol_size == other.symbol_size
            && self.max_source_symbols == other.max_source_symbols
            && match (&self.closed, &other.closed) {
                (None, None) => self.slab == other.slab,
                (Some(a), Some(b)) => {
                    // C is a function of the k source symbols, and the
                    // plan's mapping decides where each of its rows sits.
                    a.source_symbols == b.source_symbols
                        && (0..num_intermediate_symbols(a.source_symbols) as usize)
                            .all(|j| self.intermediate(a, j) == other.intermediate(b, j))
                }
                _ => false,
            }
    }
}

impl Eq for InPlaceBlockEncoder {}

impl InPlaceBlockEncoder {
    /// An open, empty encoder for blocks of up to `max_source_symbols` (K)
    /// symbols of `symbol_size` (T) bytes. `full_plan` must be a plan for
    /// K'(K), typically [`SourceBlockEncodingPlan::cached`] of K; it encodes
    /// every block whose K' is K'(K). This allocates the L(K) * T slab.
    ///
    /// # Errors
    /// - [`BlockError::InvalidSourceSymbols`] if K is zero or above
    ///   `MAX_SOURCE_SYMBOLS_PER_BLOCK`;
    /// - [`BlockError::InvalidConfig`] if T is zero or `full_plan` is not a
    ///   plan for K'(K).
    pub fn new(
        symbol_size: u16,
        max_source_symbols: u32,
        full_plan: Arc<SourceBlockEncodingPlan>,
    ) -> Result<Self, BlockError> {
        check_source_symbols(max_source_symbols)?;
        let extended = extended_source_block_symbols(max_source_symbols);
        if symbol_size == 0
            || extended_source_block_symbols(u32::from(full_plan.source_symbol_count())) != extended
        {
            return Err(BlockError::InvalidConfig);
        }
        let symbol_size = usize::from(symbol_size);
        Ok(InPlaceBlockEncoder {
            symbol_size,
            max_source_symbols,
            offset: front_rows(max_source_symbols),
            slab: SymbolSlab::with_zeros(
                num_intermediate_symbols(max_source_symbols) as usize,
                symbol_size,
            ),
            full_plan,
            closed: None,
        })
    }

    /// A closed encoder for one source block of `source_symbols` (k)
    /// symbols, built from its bytes: `data` is the block as laid out in the
    /// object (RFC 6330 section 4.4.1.2, sub-blocks per `config`), zero
    /// extended to k * T. `plan` must be a plan for K'(k).
    ///
    /// Each byte of `data` is written once, straight into its row of D; the
    /// rest of D (its S + H and K' - k rows, and the zero extension) is
    /// zeroed once. Nothing is calloc'd and no source copy is kept. The
    /// encoder is equivalent to one built with [`new`](Self::new) of k,
    /// filled and closed at k.
    ///
    /// # Errors
    /// - [`BlockError::InvalidSourceSymbols`] if k is zero or above
    ///   `MAX_SOURCE_SYMBOLS_PER_BLOCK`;
    /// - [`BlockError::InvalidConfig`] if `config` cannot describe a source
    ///   block or `plan` is not a plan for K'(k);
    /// - [`BlockError::DataTooLong`] if `data` is longer than k * T.
    pub fn from_source_block(
        config: &ObjectTransmissionInformation,
        source_symbols: u32,
        data: &[u8],
        plan: Arc<SourceBlockEncodingPlan>,
    ) -> Result<Self, BlockError> {
        check_source_symbols(source_symbols)?;
        let layout = SubBlockLayout::new(config)?;
        let extended = extended_source_block_symbols(source_symbols);
        if extended_source_block_symbols(u32::from(plan.source_symbol_count())) != extended {
            return Err(BlockError::InvalidConfig);
        }
        let t = layout.symbol_size();
        let k = source_symbols as usize;
        let front = front_rows(source_symbols);
        let rows = num_intermediate_symbols(source_symbols) as usize;
        let (Some(block_len), Some(slab_len)) = (k.checked_mul(t), rows.checked_mul(t)) else {
            return Err(BlockError::InvalidConfig);
        };
        if data.len() > block_len {
            return Err(BlockError::DataTooLong {
                len: data.len(),
                max: block_len,
            });
        }
        let mut bytes = Vec::with_capacity(slab_len);
        {
            let (head, rest) = bytes.spare_capacity_mut()[..slab_len].split_at_mut(front * t);
            let (source, tail) = rest.split_at_mut(block_len);
            zero(head);
            zero(tail);
            let mut written = 0;
            layout.for_each_run(k, 0, data.len(), |from, to, len| {
                let dest = &mut source[to..to + len];
                // SAFETY: `dest` is `len` writable bytes that do not overlap
                // `data`, which is borrowed immutably.
                unsafe {
                    ptr::copy_nonoverlapping(data[from..].as_ptr(), dest.as_mut_ptr().cast(), len)
                };
                written += len;
            });
            layout.for_each_run(k, data.len(), block_len, |_, to, len| {
                zero(&mut source[to..to + len]);
                written += len;
            });
            // The runs of [0, k * T) cover every source byte exactly once.
            assert_eq!(written, block_len);
        }
        // SAFETY: the head, every source byte and the tail were initialized
        // above, and together they are the first `slab_len` bytes.
        unsafe { bytes.set_len(slab_len) };
        let mut slab = SymbolSlab::from_bytes(bytes, t);
        replay(row_operations(&plan), 0, &mut slab);
        Ok(InPlaceBlockEncoder {
            symbol_size: t,
            max_source_symbols: source_symbols,
            offset: front,
            slab,
            full_plan: Arc::clone(&plan),
            closed: Some(Closed {
                source_symbols,
                base: 0,
                plan,
            }),
        })
    }

    /// The symbol size T in bytes.
    pub fn symbol_size(&self) -> u16 {
        self.symbol_size as u16
    }

    /// K, the most source symbols a block may hold.
    pub fn max_source_symbols(&self) -> u32 {
        self.max_source_symbols
    }

    /// The closed block's source symbol count k, or `None` while open.
    pub fn closed_source_symbols(&self) -> Option<u32> {
        self.closed.as_ref().map(|c| c.source_symbols)
    }

    /// Start a new block, dropping the closed one if any. Source rows keep
    /// stale bytes until the caller writes them; only rows below the k given
    /// to [`close`](Self::close) are read.
    pub fn reopen(&mut self) {
        self.closed = None;
    }

    /// The D row of source symbol `index`, T bytes, for the caller to fill.
    /// `None` when a block is closed or `index >= K`.
    pub fn source_row_mut(&mut self, index: u32) -> Option<&mut [u8]> {
        if self.closed.is_some() || index >= self.max_source_symbols {
            return None;
        }
        Some(self.slab.get_mut(self.offset + index as usize))
    }

    /// The D row of source symbol `index` as last written. `None` when a
    /// block is closed or `index >= K`.
    pub fn source_row(&self, index: u32) -> Option<&[u8]> {
        if self.closed.is_some() || index >= self.max_source_symbols {
            return None;
        }
        Some(self.slab.get(self.offset + index as usize))
    }

    /// Close the block as an RFC 6330 source block of its first
    /// `source_symbols` (k) source rows, each of which must have been written
    /// since [`reopen`](Self::reopen), and compute its intermediate symbols
    /// in place. The plan for K'(k) comes from the full-block plan or from
    /// `memo` (see the type documentation). Closing an already closed block
    /// with the same k does nothing.
    ///
    /// # Errors
    /// [`BlockError::InvalidSourceSymbols`] if k is zero, above K, or
    /// differs from the k of an already closed block. On error nothing
    /// changes.
    pub fn close(
        &mut self,
        source_symbols: u32,
        memo: &mut EncodingPlanMemo,
    ) -> Result<(), BlockError> {
        if source_symbols == 0 || source_symbols > self.max_source_symbols {
            return Err(BlockError::InvalidSourceSymbols(source_symbols));
        }
        if let Some(closed) = &self.closed {
            return if closed.source_symbols == source_symbols {
                Ok(())
            } else {
                Err(BlockError::InvalidSourceSymbols(source_symbols))
            };
        }
        let k = source_symbols as usize;
        let extended = extended_source_block_symbols(source_symbols) as usize;
        let base = self.offset - front_rows(source_symbols);
        let t = self.symbol_size;
        {
            // The window's S + H rows, then its K' - k padding rows. The k
            // source rows between them were written at admission.
            let bytes = self.slab.as_bytes_mut();
            bytes[base * t..self.offset * t].fill(0);
            bytes[(self.offset + k) * t..(self.offset + extended) * t].fill(0);
        }
        let plan = if extended == extended_source_block_symbols(self.max_source_symbols) as usize {
            Arc::clone(&self.full_plan)
        } else {
            memo.plan_for(source_symbols)
        };
        replay(row_operations(&plan), base, &mut self.slab);
        self.closed = Some(Closed {
            source_symbols,
            base,
            plan,
        });
        Ok(())
    }

    /// Write the `count` encoding symbols of the closed block with ESIs
    /// `first_esi`, `first_esi + 1`, ... into `out`, symbol `i` at
    /// `out[i * stride..i * stride + T]`. Other bytes of `out` are left
    /// untouched, so `out` may be uninitialized and the caller can reserve
    /// per-symbol headers with a stride of header + T.
    ///
    /// An ESI below k gives that source symbol (computed from C, section
    /// 5.3.3.4.1), an ESI of k or more repair symbol `esi - k` (internal
    /// symbol ID `esi + K' - k`, section 5.3.2).
    ///
    /// # Errors
    /// [`BlockError::NoBlock`] if no block is closed,
    /// [`BlockError::InvalidStride`] if `stride < T`,
    /// [`BlockError::InvalidEsi`] if the range reaches 2^24, and
    /// [`BlockError::BufferTooSmall`] if `out` is shorter than
    /// `(count - 1) * stride + T`. On error nothing is written.
    pub fn symbols_into(
        &self,
        first_esi: u32,
        count: u32,
        out: &mut [MaybeUninit<u8>],
        stride: usize,
    ) -> Result<(), BlockError> {
        let Some(closed) = &self.closed else {
            return Err(BlockError::NoBlock);
        };
        let t = self.symbol_size;
        let needed = strided_len(count as usize, t, stride)?;
        if out.len() < needed {
            return Err(BlockError::BufferTooSmall {
                len: out.len(),
                needed,
            });
        }
        if count > 0 && u64::from(first_esi) + u64::from(count) > u64::from(ESI_LIMIT) {
            return Err(BlockError::InvalidEsi(first_esi.max(ESI_LIMIT)));
        }
        let k = closed.source_symbols;
        let padding = extended_source_block_symbols(k) - k;
        let (lt, j, p1) = (num_lt_symbols(k), systematic_index(k), calculate_p1(k));
        for i in 0..count {
            let esi = first_esi + i;
            let isi = if esi < k { esi } else { esi + padding };
            let start = i as usize * stride;
            enc_into_uninit(
                &mut out[start..start + t],
                k,
                |row| self.intermediate(closed, row),
                intermediate_tuple(isi, lt, j, p1),
            );
        }
        Ok(())
    }

    /// Intermediate symbol `j` (C[j]) of the closed block.
    fn intermediate(&self, closed: &Closed, j: usize) -> &[u8] {
        self.slab.get(closed.base + reorder(&closed.plan)[j])
    }
}

/// Zero uninitialized bytes.
fn zero(bytes: &mut [MaybeUninit<u8>]) {
    // SAFETY: `bytes` is `bytes.len()` writable bytes.
    unsafe { ptr::write_bytes(bytes.as_mut_ptr(), 0, bytes.len()) };
}

/// S + H of K'(k): the rows of D in front of the source symbols.
fn front_rows(source_symbols: u32) -> usize {
    (num_ldpc_symbols(source_symbols) + num_hdpc_symbols(source_symbols)) as usize
}

/// A plan's row operations: everything before its final reorder.
fn row_operations(plan: &SourceBlockEncodingPlan) -> &[SymbolOps] {
    let (last, ops) = plan
        .operations()
        .split_last()
        .expect("a plan ends with its reorder");
    assert!(matches!(last, SymbolOps::Reorder { .. }));
    ops
}

/// A plan's final reorder: intermediate symbol `j` is D row `order[j]`.
fn reorder(plan: &SourceBlockEncodingPlan) -> &[usize] {
    match plan.operations().last() {
        Some(SymbolOps::Reorder { order }) => order,
        _ => unreachable!("a plan ends with its reorder"),
    }
}

/// Replay row operations over the window of `slab` that starts at row
/// `base`.
fn replay(ops: &[SymbolOps], base: usize, slab: &mut SymbolSlab) {
    for op in ops {
        match op {
            SymbolOps::AddAssign { dest, src } => slab.add_assign(base + dest, base + src),
            SymbolOps::MulAssign { dest, scalar } => slab.mulassign_scalar(base + dest, scalar),
            SymbolOps::FMA { dest, src, scalar } => slab.fma(base + dest, base + src, scalar),
            SymbolOps::Reorder { .. } => unreachable!("only the final operation reorders"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::systematic_constants::{
        SYSTEMATIC_INDICES_AND_PARAMETERS, extended_source_block_symbols,
    };

    use super::front_rows;

    /// The window of a k-symbol block fits the slab of a K-symbol encoder
    /// because S + H never decreases as K' grows.
    #[test]
    fn front_rows_never_decrease() {
        let mut previous = 0;
        for &(extended, ..) in SYSTEMATIC_INDICES_AND_PARAMETERS.iter() {
            let rows = front_rows(extended);
            assert!(rows >= previous, "K'={extended}");
            assert_eq!(extended_source_block_symbols(extended), extended);
            previous = rows;
        }
    }
}
