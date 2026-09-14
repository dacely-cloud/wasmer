//! Linear-memory mmap pooling — REMOVED.
//!
//! This module used to recycle anonymous `MAP_PRIVATE` mappings across wasm
//! `Instance`s (a per-thread pool keyed by `(accessible_size, mapping_size)`)
//! to avoid `mmap`/`munmap` per instantiation. Reuse turned out to be unsound:
//! recycling a mapping across Instances aliases base pointers between tenants,
//! and on the async dispatch path a tenant's `memory.grow` (run across a
//! coroutine/host-stack switch) writes `current_length` through a
//! `VMMemoryDefinition` that a *different* instance reusing the mapping then
//! reads — the victim takes the grown reset path and writes through its
//! read-only soft guard → SIGSEGV / cross-tenant state. Root-caused with rr
//! under multi-tenant async.
//!
//! The pool is gone. Every linear-memory mapping is `mmap`'d fresh and
//! `munmap`'d on drop. Instances are still pooled at the host `PooledInstance`
//! layer, so this only costs an `mmap` at instance *build*, not per request.

use crate::mmap::Mmap;

/// No pool: there is never a recycled mapping, so the caller always `mmap`s a
/// fresh one.
#[inline]
pub(crate) fn try_take(_accessible_size: usize, _mapping_size: usize) -> Option<Mmap> {
    None
}

/// No pool: hand the mapping straight back so the caller `munmap`s it on drop.
#[inline]
pub(crate) fn try_pool(mmap: Mmap) -> Option<Mmap> {
    Some(mmap)
}
