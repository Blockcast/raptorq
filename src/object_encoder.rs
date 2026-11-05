#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec::Vec};
use core::mem::MaybeUninit;
#[cfg(feature = "std")]
use std::{sync::Arc, vec::Vec};

use crate::base::{ObjectTransmissionInformation, PayloadId, partition};
use crate::encoder::SourceBlockEncodingPlan;
#[cfg(feature = "std")]
use crate::inplace_encoder::stored_plan;
use crate::inplace_encoder::{EncodingPlanMemo, InPlaceBlockEncoder};
use crate::reusable::{BlockError, check_source_symbols, strided_len};
use crate::util::int_div_ceil;

/// An RFC 6330 object encoder that builds each source block's D vector in
/// place: every source byte is written once, into its block's
/// [`InPlaceBlockEncoder`], and symbols are written straight into caller
/// buffers through [`InPlaceBlockEncoder::symbols_into`].
///
/// It partitions the object exactly as [`Encoder::new`](crate::Encoder::new)
/// does and produces the same symbols. Plans for K' <= 257 come from a
/// process-wide store keyed by K' (generated once per process for each K',
/// at most 46 plans), so a warm encoder generates no plan. Above that, and
/// without the `std` feature, the encoder generates one plan per distinct K'
/// in the object and keeps it for its lifetime: its blocks address their
/// intermediate symbols through it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectEncoder {
    config: ObjectTransmissionInformation,
    blocks: Vec<InPlaceBlockEncoder>,
    source_symbols: u32,
}

impl ObjectEncoder {
    /// Encode `data` as the object `config` describes. Bytes past the
    /// transfer length are ignored, and the last symbol of the object is
    /// zero extended.
    ///
    /// # Errors
    /// - [`BlockError::InvalidConfig`] if `config` cannot describe a source
    ///   block or has no source blocks;
    /// - [`BlockError::InvalidSourceSymbols`] if a block would have zero or
    ///   more than `MAX_SOURCE_SYMBOLS_PER_BLOCK` symbols.
    pub fn new(data: &[u8], config: ObjectTransmissionInformation) -> Result<Self, BlockError> {
        let t = usize::from(config.symbol_size());
        if t == 0 || config.source_blocks() == 0 {
            return Err(BlockError::InvalidConfig);
        }
        let data = &data[..data.len().min(config.transfer_length() as usize)];
        let total = int_div_ceil(config.transfer_length(), u64::from(config.symbol_size()));
        let (kl, ks, zl, zs) = partition(total, config.source_blocks());
        check_source_symbols(kl)?;
        if zs > 0 {
            check_source_symbols(ks)?;
        }
        // Plans above the store's K' limit, kept for this object's blocks.
        let mut kept = EncodingPlanMemo::new(usize::MAX);
        let mut blocks = Vec::with_capacity((zl + zs) as usize);
        let mut start = 0usize;
        for block in 0..zl + zs {
            let k = if block < zl { kl } else { ks };
            let end = start + k as usize * t;
            let bytes = &data[start.min(data.len())..end.min(data.len())];
            let plan = plan_for(k, &mut kept);
            blocks.push(InPlaceBlockEncoder::from_source_block(
                &config, k, bytes, plan,
            )?);
            start = end;
        }
        Ok(ObjectEncoder {
            config,
            blocks,
            source_symbols: total,
        })
    }

    /// The object's transmission information.
    pub fn config(&self) -> ObjectTransmissionInformation {
        self.config
    }

    /// The object's source symbols over all blocks (Kt).
    pub fn source_symbol_count(&self) -> u32 {
        self.source_symbols
    }

    /// The source blocks, by source block number. Each is closed: its
    /// [`closed_source_symbols`](InPlaceBlockEncoder::closed_source_symbols)
    /// is its k, and its symbols come from
    /// [`symbols_into`](InPlaceBlockEncoder::symbols_into) (ESI k + r is
    /// repair symbol r).
    pub fn blocks(&self) -> &[InPlaceBlockEncoder] {
        &self.blocks
    }

    /// Write `count` serialized [`EncodingPacket`](crate::EncodingPacket)s
    /// of source block `block`, ESIs `first_esi`, `first_esi + 1`, ..., into
    /// `out`, back to back: each record is its 4-byte
    /// [`PayloadId`] then its T-byte symbol, exactly the bytes of
    /// `EncodingPacket::serialize`. ESIs below k are source packets, ESI
    /// k + r repair symbol r. Returns the bytes written, `count * (4 + T)`.
    /// `out` may be uninitialized; nothing past the records is touched.
    ///
    /// # Errors
    /// [`BlockError::NoBlock`] if there is no block `block`, and the errors
    /// of [`InPlaceBlockEncoder::symbols_into`] (with the record stride). On
    /// error nothing is written.
    pub fn packets_into(
        &self,
        block: usize,
        first_esi: u32,
        count: u32,
        out: &mut [MaybeUninit<u8>],
    ) -> Result<usize, BlockError> {
        let encoder = self.blocks.get(block).ok_or(BlockError::NoBlock)?;
        let record = 4 + usize::from(self.config.symbol_size());
        let len = strided_len(count as usize, record, record)?;
        if count == 0 {
            return Ok(0);
        }
        if out.len() < len {
            return Err(BlockError::BufferTooSmall {
                len: out.len(),
                needed: len,
            });
        }
        encoder.symbols_into(first_esi, count, &mut out[4..len], record)?;
        for (i, header) in out[..len].chunks_exact_mut(record).enumerate() {
            let id = PayloadId::new(block as u8, first_esi + i as u32).serialize();
            for (dst, byte) in header.iter_mut().zip(id) {
                dst.write(byte);
            }
        }
        Ok(len)
    }
}

fn plan_for(source_symbols: u32, kept: &mut EncodingPlanMemo) -> Arc<SourceBlockEncodingPlan> {
    #[cfg(feature = "std")]
    if let Some(plan) = stored_plan(source_symbols) {
        return plan;
    }
    kept.plan_for(source_symbols)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use crate::inplace_encoder::{STORE_BYTES, plan_store_usage, stored_plan};
    use crate::systematic_constants::SYSTEMATIC_INDICES_AND_PARAMETERS;

    /// The store holds a plan for every K' up to 257 within its budget.
    #[test]
    fn plan_store_holds_every_small_extended_size() {
        let mut count = 0;
        for &(extended, ..) in SYSTEMATIC_INDICES_AND_PARAMETERS.iter() {
            if extended > 257 {
                break;
            }
            count += 1;
            let a = stored_plan(extended).unwrap();
            let b = stored_plan(extended).unwrap();
            assert!(std::sync::Arc::ptr_eq(&a, &b), "K'={extended} not kept");
        }
        assert!(stored_plan(258).is_none());
        let (bytes, plans) = plan_store_usage();
        assert_eq!((plans, count), (46, 46));
        assert!(bytes <= STORE_BYTES, "{bytes}");
        #[cfg(target_pointer_width = "64")]
        assert_eq!(bytes, STORE_BYTES);
        std::println!("plan store: {plans} plans, {bytes} B");
    }
}
