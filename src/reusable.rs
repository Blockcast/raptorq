//! Shared pieces of the reusable block codec API:
//! [`ReusableSourceBlockEncoder`](crate::ReusableSourceBlockEncoder) and
//! [`ReusableSourceBlockDecoder`](crate::ReusableSourceBlockDecoder).

use core::fmt;

use crate::base::ObjectTransmissionInformation;
use crate::base::partition;
use crate::systematic_constants::MAX_SOURCE_SYMBOLS_PER_BLOCK;

/// Encoding symbol IDs are 24-bit (RFC 6330 section 3.2): every ESI must be
/// below this.
pub(crate) const ESI_LIMIT: u32 = 1 << 24;

/// The error returned by the reusable block codec.
///
/// Every method that returns an error validates its arguments before it
/// changes anything: on `Err` the encoder or decoder, and any output buffer,
/// are exactly as they were before the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BlockError {
    /// The configuration cannot describe a source block: the symbol size,
    /// symbol alignment or sub-block count is zero, or the symbol size is not
    /// a multiple of the alignment.
    InvalidConfig,
    /// The source symbol count is zero or above
    /// [`MAX_SOURCE_SYMBOLS_PER_BLOCK`](crate::MAX_SOURCE_SYMBOLS_PER_BLOCK).
    InvalidSourceSymbols(u32),
    /// No source block is loaded: the encoder has not encoded one since it
    /// was created or cleared, or the decoder has not been reset to one.
    NoBlock,
    /// The source data is longer than `source_symbols * symbol_size`.
    DataTooLong { len: usize, max: usize },
    /// A received symbol is not exactly `symbol_size` bytes.
    SymbolSizeMismatch { len: usize, expected: usize },
    /// An encoding symbol ID is not below 2^24.
    InvalidEsi(u32),
    /// The output stride is smaller than the symbol size.
    InvalidStride { stride: usize, symbol_size: usize },
    /// The output buffer is shorter than the operation needs.
    BufferTooSmall { len: usize, needed: usize },
    /// The requested byte range extends past the end of the source block.
    OutOfRange {
        offset: usize,
        len: usize,
        block_len: usize,
    },
    /// The decoder has not recovered the source block yet.
    NotDecoded,
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            BlockError::InvalidConfig => write!(f, "invalid symbol size, alignment or sub-blocks"),
            BlockError::InvalidSourceSymbols(k) => write!(
                f,
                "invalid source symbol count {k} (must be 1..={MAX_SOURCE_SYMBOLS_PER_BLOCK})"
            ),
            BlockError::NoBlock => write!(f, "no source block is loaded"),
            BlockError::DataTooLong { len, max } => {
                write!(f, "source data is {len} bytes, more than the block's {max}")
            }
            BlockError::SymbolSizeMismatch { len, expected } => {
                write!(f, "symbol is {len} bytes, expected {expected}")
            }
            BlockError::InvalidEsi(esi) => write!(f, "encoding symbol ID {esi} is not below 2^24"),
            BlockError::InvalidStride {
                stride,
                symbol_size,
            } => write!(
                f,
                "stride {stride} is smaller than the symbol size {symbol_size}"
            ),
            BlockError::BufferTooSmall { len, needed } => {
                write!(f, "output buffer is {len} bytes, need {needed}")
            }
            BlockError::OutOfRange {
                offset,
                len,
                block_len,
            } => write!(
                f,
                "byte range {offset}+{len} extends past the {block_len}-byte source block"
            ),
            BlockError::NotDecoded => write!(f, "the source block has not been decoded"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BlockError {}

pub(crate) fn check_source_symbols(source_symbols: u32) -> Result<(), BlockError> {
    if source_symbols == 0 || source_symbols > MAX_SOURCE_SYMBOLS_PER_BLOCK {
        return Err(BlockError::InvalidSourceSymbols(source_symbols));
    }
    Ok(())
}

/// The bytes needed to write `count` symbols of `symbol_size` bytes,
/// `stride` bytes apart.
pub(crate) fn strided_len(
    count: usize,
    symbol_size: usize,
    stride: usize,
) -> Result<usize, BlockError> {
    if stride < symbol_size {
        return Err(BlockError::InvalidStride {
            stride,
            symbol_size,
        });
    }
    if count == 0 {
        return Ok(0);
    }
    (count - 1)
        .checked_mul(stride)
        .and_then(|n| n.checked_add(symbol_size))
        .ok_or(BlockError::BufferTooSmall {
            len: 0,
            needed: usize::MAX,
        })
}

/// How a source block's bytes map onto its symbols (RFC 6330 section
/// 4.4.1.2). With N sub-blocks each symbol is the concatenation of N
/// sub-symbols, and the block is laid out sub-block by sub-block: sub-block
/// `j` holds sub-symbol `j` of symbol 0, then of symbol 1, and so on. With
/// one sub-block the block is just its symbols in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SubBlockLayout {
    symbol_size: usize,
    /// Number of long sub-blocks, and the bytes each one contributes per
    /// symbol; then the same for the short sub-blocks.
    long_count: usize,
    long_bytes: usize,
    short_count: usize,
    short_bytes: usize,
}

impl SubBlockLayout {
    pub(crate) fn new(config: &ObjectTransmissionInformation) -> Result<Self, BlockError> {
        let symbol_size = config.symbol_size();
        let alignment = config.symbol_alignment();
        let sub_blocks = config.sub_blocks();
        if symbol_size == 0
            || alignment == 0
            || !symbol_size.is_multiple_of(u16::from(alignment))
            || sub_blocks == 0
        {
            return Err(BlockError::InvalidConfig);
        }
        let (tl, ts, nl, ns) = partition(u32::from(symbol_size / u16::from(alignment)), sub_blocks);
        Ok(SubBlockLayout {
            symbol_size: usize::from(symbol_size),
            long_count: nl as usize,
            long_bytes: tl as usize * usize::from(alignment),
            short_count: ns as usize,
            short_bytes: ts as usize * usize::from(alignment),
        })
    }

    #[inline]
    pub(crate) fn symbol_size(&self) -> usize {
        self.symbol_size
    }

    #[inline]
    pub(crate) fn is_contiguous(&self) -> bool {
        self.long_count + self.short_count == 1
    }

    /// `(offset within each symbol, bytes per symbol)` of every sub-block.
    fn sub_blocks(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let long = (0..self.long_count).map(move |j| (j * self.long_bytes, self.long_bytes));
        let base = self.long_count * self.long_bytes;
        let short =
            (0..self.short_count).map(move |j| (base + j * self.short_bytes, self.short_bytes));
        long.chain(short)
    }

    /// Call `f(block_offset, symbol_offset, len)` for each run of bytes that
    /// is contiguous in both the block and the symbol slab, over the block
    /// byte range `[start, end)` of a block of `source_symbols` symbols.
    /// `symbol_offset` is a byte offset into the concatenated symbols.
    pub(crate) fn for_each_run(
        &self,
        source_symbols: usize,
        start: usize,
        end: usize,
        mut f: impl FnMut(usize, usize, usize),
    ) {
        if start >= end {
            return;
        }
        if self.is_contiguous() {
            f(start, start, end - start);
            return;
        }
        for (sym_offset, bytes) in self.sub_blocks() {
            if bytes == 0 {
                continue;
            }
            let block_base = sym_offset * source_symbols;
            let block_limit = block_base + bytes * source_symbols;
            if block_limit <= start || block_base >= end {
                continue;
            }
            let lo = start.max(block_base);
            let hi = end.min(block_limit);
            let mut pos = lo;
            while pos < hi {
                let rel = pos - block_base;
                let symbol = rel / bytes;
                let within = rel % bytes;
                let len = (bytes - within).min(hi - pos);
                f(pos, symbol * self.symbol_size + sym_offset + within, len);
                pos += len;
            }
        }
    }
}
