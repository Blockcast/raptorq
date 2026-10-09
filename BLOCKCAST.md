# Blockcast fork of raptorq

This is Blockcast's fork of [cberner/raptorq](https://github.com/cberner/raptorq).
The `blockcast` branch (the default branch) is upstream `master` plus a short
patch stack. `master` mirrors upstream and carries no Blockcast changes.

The fork replaces the copy of raptorq that used to be vendored into
`Blockcast/fec-raptorq` at `internal/raptorq` (upstream e777861, v2.0.0, plus
local edits). Keep Blockcast changes here as commits on `blockcast`, not as
edits inside a consumer.

## Upstream base

| | |
|---|---|
| Upstream | https://github.com/cberner/raptorq |
| Base | `83cf194` "Update pyo3 to 0.29.0" (upstream `master`, v2.0.1 + 3) |
| Previous vendored base | `e777861` (v2.0.0) |

## Patches on top of upstream

In order, oldest first:

1. **octets: check every kernel against GF(256) arithmetic at lengths 0..=300.**
   Test only. It runs every length from 0 to 300 and every scalar through each
   dispatched kernel, so it covers whichever SIMD path the target selects.
   It is a candidate for upstreaming.
2. **octets: add WebAssembly SIMD128 kernels** (`src/octets_simd128.rs`). This
   ports kkroo/fec-raptorq `5842ded` and `3433992`. wasm32 has no runtime
   feature detection, so the kernels are compiled in only with
   `-C target-feature=+simd128` (and the `std` feature). Without that flag the
   scalar path is used, as upstream does.
3. **lib: export `MAX_SOURCE_SYMBOLS_PER_BLOCK`** (from kkroo/fec-raptorq
   `ec2334b`). The fec-raptorq C bindings use it to reject an oversized block
   before the encoder asserts.
4. This file.
5. **Reusable block codec** (`src/reusable*.rs`): `ReusableSourceBlockEncoder`,
   `ReusableSourceBlockDecoder` and `BlockError`. This API is additive: the
   existing types are untouched. An encoder is refilled block after block,
   and a decoder is reset block after block, keeping their storage. Symbols
   go into caller buffers (`repair_into`, `repair_range_into`,
   `copy_block_into`). The decoder deduplicates by ESI and does not solve
   again until a new symbol arrives. It is the crate side of the fec-raptorq
   zero-allocation batched FFI. Tests: `tests/reusable_codec.rs`
   (differential and reuse) and `tests/reusable_alloc.rs` (steady-state
   allocation counts). Supporting changes: `SourceBlockEncodingPlan::cached(K)`
   (upstream's process-wide plan cache) and `source_symbol_count()`, plus
   crate-private `SymbolSlab` helpers that reshape a slab in place.
   The encoder encodes every block by replaying a plan for the block's
   extended size K' into its own storage (a plan depends only on K', so one
   plan serves every K with that K'). Plans come from the attached plan or
   from a per-encoder memo that keeps the plan it generates for each new K'
   (byte budget, 4 MiB by default, never evicts; the process-wide cache is
   not touched). Encoding and symbol generation therefore allocate nothing
   once each K' has been seen and its plan is kept, including the short
   final block of every object, as long as the memo's budget holds those
   plans. The default 4 MiB (`DEFAULT_PLAN_MEMO_BYTES`) holds plans for every
   K' up to 217 together, which covers every short block of a stream of
   blocks of up to 218 symbols; a stream of 128-symbol blocks needs 27 plans,
   about 1.7 MB. Past the budget, a K' whose plan was not kept is solved again
   on every block, which allocates; `reserve` returns `false` for it.
   Decoding allocates nothing only when no symbol is lost: a decode that has
   to solve still runs the allocating RFC 6330 solver
   (`decode_with_loss_does_not_allocate` is `#[ignore]`d).
   `benches/reusable_encode_benchmark.rs` times encode + repair at K=128,
   T=1344.

The old vendored copy also had some dead-code removals in `arraymap.rs`,
`matrix.rs`, `sparse_matrix.rs` and `sparse_vec.rs`: `size_in_bytes` was gated
behind `benchmarking`, and `OctetIter` lifetimes were made explicit. They are
not carried here. Upstream fixed the same warnings itself in `ccf9f73`,
and the tree builds without warnings on aarch64, with
`--no-default-features`, and on wasm32 with and without SIMD128.

## Syncing upstream

```sh
git remote add upstream https://github.com/cberner/raptorq.git   # once
git fetch upstream
git checkout master && git merge --ff-only upstream/master && git push origin master
git checkout blockcast && git rebase upstream/master
cargo test --release
cargo build --no-default-features
cargo build --release --target wasm32-unknown-unknown
RUSTFLAGS="-C target-feature=+simd128" cargo build --release --target wasm32-unknown-unknown
git push --force-with-lease origin blockcast
```

After the rebase, update the base revision in the table above. `blockcast` is
a rebased patch stack, so it is force-pushed (with lease) on every sync.
Consumers pin a commit, never the branch name. A pinned commit can stop being
reachable from `blockcast` after a sync, so tag each commit that a consumer
pins (`blockcast-YYYYMMDD`) to keep it fetchable.

The octets tests cover the SIMD128 kernels only when they run on wasm.
`cargo test` does not run there because criterion's rayon does not build for
wasi. To run them, copy the crate without the `[[bench]]` targets and the
`criterion` dev-dependency, then run
`cargo test --release --lib --target wasm32-wasip1` with
`RUSTFLAGS="-C target-feature=+simd128"` and a WASI runner (wasmtime, or
node's `node:wasi`). The `decoder::codec_tests::repair*` tests use threads,
which wasi lacks, so run them serially or skip them.

## Consumers

All three live in [Blockcast/fec-raptorq](https://github.com/Blockcast/fec-raptorq),
which `Blockcast/multicast` uses as the `fec-raptorq` submodule:

| Crate | Uses | Features |
|---|---|---|
| `c-bindings` | the cgo FFI used by multicast (`raptorq_rust` build tag) | `std` |
| `wasm` | browser/Node bindings, built with `+simd128` | `default-features = false`, `std` |
| `internal` | the CLI/test binary | `std`, `serde_support` |

The upstream test suite plus a differential harness confirm that the fork
produces byte-identical source and repair packets to the old vendored copy
(e777861 + patches). The harness covered every K in 1..=300 and
K in {1000, 4096, 8192, 56403}, at T in {8, 1344, 1345}, with and without
`SourceBlockEncodingPlan`. Decoding matched too: success and output were
identical across 6000 random erasure patterns, and object round trips with
several source blocks and sub-blocks agreed. The harness ran natively on
aarch64 and on wasm32 with and without SIMD128.
