//! Snapshot and restore of a warm instance's mutable state.
//!
//! Captures the post-initialization baseline of an [`Instance`] — its defined
//! linear memories, globals, and tables — so the same instance can be reset and
//! reused for many runs instead of being re-instantiated. Re-instantiation
//! re-allocates memories, re-links imports, and re-runs the start function;
//! resetting to a snapshot skips all of that.
//!
//! This is the eager, portable path: snapshot copies each memory image into a
//! `Vec<u8>`, and restore copies it back. A Linux `memfd` + `madvise(MADV_DONTNEED)`
//! copy-on-write fast path can later be layered behind the same API to make
//! restore cost O(dirty pages) instead of O(heap size).

use std::sync::Arc;

use wasmer_types::{ElemIndex, MemoryError};

use super::{Instance, VMInstance};
use crate::table::RawTableElement;
use crate::{CowBacking, LinearMemory, VMFuncRef};

/// How a single linear memory's baseline is captured.
enum MemorySnapshot {
    /// Eager byte image (portable; used for dynamic memories, shared memories,
    /// non-Linux targets, or when a `memfd` could not be created).
    ///
    /// `dirty_prefix` is the image's highest non-zero byte, so `image[dirty_prefix..]`
    /// is all zero. The bounded restore copies only `[0, dirty_prefix)` and
    /// zeroes the rest with cache-bypassing stores instead of streaming zeros
    /// through the cache.
    Eager { image: Vec<u8>, dirty_prefix: usize },
    /// Copy-on-write backing: restore re-maps the live memory over a `memfd`,
    /// discarding dirtied pages in a single syscall (O(dirty pages)).
    Cow(CowBacking),
}

/// Whether copy-on-write acceleration is disabled via the environment. Provides
/// an operational escape hatch (`WASMER_DISABLE_INSTANCE_COW=1`) that forces the
/// eager path without changing observable behavior — both paths restore to the
/// same bytes.
fn cow_disabled() -> bool {
    std::env::var_os("WASMER_DISABLE_INSTANCE_COW").is_some()
}

/// An immutable snapshot of an instance's mutable state, captured by
/// [`VMInstance::snapshot`] and restored by [`VMInstance::reset_to_snapshot`].
///
/// The snapshot owns copies of every defined memory image, global value, and
/// table image. It is only meaningful for the instance it was taken from, and
/// the caller must keep that instance alive for the snapshot's lifetime: table
/// images hold raw func/extern refs that point into the instance.
pub struct VMInstanceSnapshot {
    /// One baseline per defined (local) linear memory, in local-index order.
    memories: Vec<MemorySnapshot>,
    /// One raw value per defined (local) global, in local-index order.
    globals: Vec<u128>,
    /// One element image per defined (local) table, in local-index order.
    tables: Vec<Vec<RawTableElement>>,
}

impl std::fmt::Debug for VMInstanceSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VMInstanceSnapshot")
            .field("memories", &self.memories.len())
            .field("globals", &self.globals.len())
            .field("tables", &self.tables.len())
            .finish()
    }
}

impl Instance {
    /// Capture this instance's defined memories, globals, and tables.
    pub(crate) fn snapshot(&self) -> VMInstanceSnapshot {
        self.snapshot_with(!cow_disabled())
    }

    /// Capture with an eager (memcpy) memory image regardless of CoW support.
    /// Required by [`restore_bounded`](Self::restore_bounded), which memcpys a
    /// prefix of the image — a CoW/memfd snapshot has no in-process byte image
    /// to slice.
    pub(crate) fn snapshot_eager(&self) -> VMInstanceSnapshot {
        self.snapshot_with(false)
    }

    fn snapshot_with(&self, allow_cow: bool) -> VMInstanceSnapshot {
        let ctx = self.context();

        let memories = self
            .memories
            .values()
            .map(|h| {
                let mem = h.get(ctx);
                // Prefer the O(dirty-pages) copy-on-write path; fall back to the
                // eager byte image if it is unsupported or a memfd could not be
                // created.
                if allow_cow
                    && mem.supports_cow_snapshot()
                    && let Ok(cow) = mem.snapshot_cow()
                {
                    return MemorySnapshot::Cow(cow);
                }
                let image = mem.snapshot_image();
                // Highest non-zero byte: everything above it is zero, so the
                // bounded restore can zero that tail with non-temporal stores
                // instead of copying it. Scanned once here, never per-reset.
                let dirty_prefix = image.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
                MemorySnapshot::Eager {
                    image,
                    dirty_prefix,
                }
            })
            .collect();

        let globals = self
            .globals
            .values()
            .map(|h| unsafe { (*h.get(ctx).vmglobal().as_ptr()).val.u128 })
            .collect();

        let tables = self
            .tables
            .values()
            .map(|h| h.get(ctx).snapshot_elements())
            .collect();

        VMInstanceSnapshot {
            memories,
            globals,
            tables,
        }
    }

    /// Restore this instance's defined memories, globals, and tables to a
    /// previously captured [`VMInstanceSnapshot`], then re-populate the passive
    /// data/element segments (undoing any `data.drop` / `elem.drop`).
    pub(crate) fn restore(&mut self, snap: &VMInstanceSnapshot) -> Result<(), MemoryError> {
        if snap.memories.len() != self.memories.len()
            || snap.globals.len() != self.globals.len()
            || snap.tables.len() != self.tables.len()
        {
            return Err(MemoryError::Generic(
                "snapshot does not match this instance's shape".into(),
            ));
        }

        {
            // SAFETY: `self.context` is valid for the instance's lifetime and is
            // not aliased while we hold this `&mut` (reset runs single-threaded
            // between wasm calls). The handle slices live in the instance, not
            // in `*self.context`, so iterating them while mutating through
            // `ctx` is disjoint — no intermediate `collect()` (restore runs
            // per request; the Vecs were measurable allocator traffic).
            let ctx = unsafe { &mut *self.context };

            for (h, mem_snap) in self.memories.values().copied().zip(&snap.memories) {
                let mem = h.get_mut(ctx);
                match mem_snap {
                    MemorySnapshot::Eager { image, .. } => mem.restore_image(image)?,
                    MemorySnapshot::Cow(cow) => mem.restore_cow(cow)?,
                }
            }
            for (h, &val) in self.globals.values().copied().zip(&snap.globals) {
                // Write through the raw definition pointer; it lives in the
                // vmctx, not in the `&VMGlobal` borrow.
                unsafe {
                    (*h.get(ctx).vmglobal().as_ptr()).val.u128 = val;
                }
            }
        }

        // Tables: restore backing elements AND re-sync the vmctx inline
        // fixed-funcref arrays that `call_indirect` dispatches through.
        self.restore_table_elements(&snap.tables);

        self.reinitialize_passive_segments();
        Ok(())
    }

    /// Like [`restore`](Self::restore), but each memory is restored only over
    /// its first `mem_dirty_bytes` (an embedder-supplied bound on the touched
    /// extent), via a plain `memcpy` — no page-table edits, so it scales across
    /// cores. The caller guarantees the tail `[mem_dirty_bytes, current_length)`
    /// is already at its reset value (e.g. held zero by a soft guard). Globals,
    /// tables, and passive segments are restored in full. Requires an eager
    /// snapshot (see [`snapshot_eager`](Self::snapshot_eager)).
    pub(crate) fn restore_bounded(
        &mut self,
        snap: &VMInstanceSnapshot,
        mem_dirty_bytes: usize,
    ) -> Result<(), MemoryError> {
        if snap.memories.len() != self.memories.len()
            || snap.globals.len() != self.globals.len()
            || snap.tables.len() != self.tables.len()
        {
            return Err(MemoryError::Generic(
                "snapshot does not match this instance's shape".into(),
            ));
        }

        {
            // SAFETY: see `restore`; single-threaded reset, no aliasing, and
            // the same disjointness argument for iterating the instance-owned
            // handle slices while mutating through `ctx`. Hot path: no
            // `collect()` allocations.
            let ctx = unsafe { &mut *self.context };
            for (h, mem_snap) in self.memories.values().copied().zip(&snap.memories) {
                let mem = h.get_mut(ctx);
                match mem_snap {
                    MemorySnapshot::Eager {
                        image,
                        dirty_prefix,
                    } => mem.restore_image_bounded(image, *dirty_prefix, mem_dirty_bytes)?,
                    MemorySnapshot::Cow(_) => {
                        return Err(MemoryError::Generic(
                            "restore_bounded requires an eager snapshot \
                             (capture with snapshot_eager)"
                                .into(),
                        ));
                    }
                }
            }
            for (h, &val) in self.globals.values().copied().zip(&snap.globals) {
                unsafe {
                    (*h.get(ctx).vmglobal().as_ptr()).val.u128 = val;
                }
            }
        }

        self.restore_table_elements(&snap.tables);
        self.reinitialize_passive_segments();
        Ok(())
    }

    /// Restore every defined table's raw elements, then re-sync the vmctx inline
    /// fixed-funcref arrays. `call_indirect` dispatches through those inline
    /// arrays, not the `VMTable` backing `vec`, so restoring the backing alone
    /// would leave a `table.set` mutation visible to indirect calls.
    fn restore_table_elements(&mut self, table_snaps: &[Vec<RawTableElement>]) {
        {
            // SAFETY: see `restore`; single-threaded reset, no aliasing.
            let ctx = unsafe { &mut *self.context };
            for (handle, elems) in self.tables.values().copied().zip(table_snaps) {
                handle.get_mut(ctx).restore_elements(elems);
            }
        }
        // `sync_fixed_funcref_table` takes `&self`, so iterating the keys
        // directly borrows cleanly — no index `collect()` on the hot path.
        for idx in self.tables.keys() {
            self.sync_fixed_funcref_table(idx);
        }
    }

    /// Reset the passive element and data segment maps back to the module's
    /// declared set, undoing any `elem.drop` / `data.drop` a run performed.
    fn reinitialize_passive_segments(&self) {
        // Nothing declared means nothing a guest `elem.drop` / `data.drop`
        // could have removed; skip the RefCell churn on the per-request
        // reset path.
        if self.module.passive_elements.is_empty() && self.module.passive_data.is_empty() {
            return;
        }
        {
            let mut passive_elements = self.passive_elements.borrow_mut();
            passive_elements.clear();
            passive_elements.extend(self.module.passive_elements.iter().filter_map(
                |(&idx, segments)| -> Option<(ElemIndex, Box<[Option<VMFuncRef>]>)> {
                    if segments.is_empty() {
                        None
                    } else {
                        Some((
                            idx,
                            segments
                                .iter()
                                .map(|s| self.func_ref(*s))
                                .collect::<Box<[Option<VMFuncRef>]>>(),
                        ))
                    }
                },
            ));
        }

        {
            let mut passive_data = self.passive_data.borrow_mut();
            passive_data.clear();
            // By-reference: one `Arc::from(&[u8])` copy per segment. The old
            // `self.module.passive_data.clone()` deep-copied every segment's
            // `Box<[u8]>` and then `Arc::from(Box)` copied it AGAIN — two full
            // byte copies of all passive data per reset.
            passive_data.extend(
                self.module
                    .passive_data
                    .iter()
                    .map(|(&idx, bytes)| (idx, Arc::from(&bytes[..]))),
            );
        }
    }
}

impl VMInstance {
    /// Capture a snapshot of this instance's mutable state (defined memories,
    /// globals, and tables) so it can later be restored with
    /// [`VMInstance::reset_to_snapshot`].
    ///
    /// Take the snapshot once the instance has reached the baseline you want to
    /// reuse (after data/element segments, the start function, and any embedder
    /// initialization). Host-side state (imported memories/globals, function
    /// environments) is not captured and must be managed by the embedder.
    pub fn snapshot(&self) -> VMInstanceSnapshot {
        self.instance().snapshot()
    }

    /// Capture a snapshot with an eager (memcpy) memory image — required for the
    /// bounded fast path [`VMInstance::reset_to_snapshot_bounded`].
    pub fn snapshot_eager(&self) -> VMInstanceSnapshot {
        self.instance().snapshot_eager()
    }

    /// Restore this instance to a previously captured [`VMInstanceSnapshot`].
    ///
    /// Returns an error if the snapshot's shape does not match this instance, or
    /// if a memory could not be grown back to the snapshot size.
    pub fn reset_to_snapshot(&mut self, snapshot: &VMInstanceSnapshot) -> Result<(), MemoryError> {
        self.instance_mut().restore(snapshot)
    }

    /// Restore from a snapshot, copying each memory only over its first
    /// `mem_dirty_bytes` (a `memcpy`, no page-table edits → scales across
    /// cores). The caller guarantees the tail beyond that bound is already at
    /// its reset value. Requires an eager snapshot
    /// ([`VMInstance::snapshot_eager`]); errors on a CoW snapshot.
    pub fn reset_to_snapshot_bounded(
        &mut self,
        snapshot: &VMInstanceSnapshot,
        mem_dirty_bytes: usize,
    ) -> Result<(), MemoryError> {
        self.instance_mut().restore_bounded(snapshot, mem_dirty_bytes)
    }
}

/// A table-only snapshot: the raw element image of every defined table, in
/// local-index order.
///
/// Unlike [`VMInstanceSnapshot`] this captures *only* tables — no linear memory
/// and no globals — so an embedder can pair its own (possibly faster) memory and
/// global reset with correct raw-element table restoration. Restoring copies the
/// raw `funcref`/`externref` slots back verbatim, with no `Value` round-trip,
/// which is exactly what reusing the *same* instance requires.
pub struct VMTablesSnapshot {
    tables: Vec<Vec<RawTableElement>>,
}

impl std::fmt::Debug for VMTablesSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VMTablesSnapshot")
            .field("tables", &self.tables.len())
            .finish()
    }
}

impl Instance {
    /// Capture the raw element image of every defined table.
    pub(crate) fn snapshot_tables(&self) -> VMTablesSnapshot {
        let ctx = self.context();
        let tables = self
            .tables
            .values()
            .map(|h| h.get(ctx).snapshot_elements())
            .collect();
        VMTablesSnapshot { tables }
    }

    /// Restore every defined table to a captured [`VMTablesSnapshot`].
    pub(crate) fn restore_tables(&mut self, snap: &VMTablesSnapshot) -> Result<(), MemoryError> {
        if snap.tables.len() != self.tables.len() {
            return Err(MemoryError::Generic(
                "tables snapshot does not match this instance's table count".into(),
            ));
        }
        self.restore_table_elements(&snap.tables);
        Ok(())
    }
}

impl VMInstance {
    /// Capture a table-only snapshot (see [`VMTablesSnapshot`]). Narrower than
    /// [`VMInstance::snapshot`]: it touches no linear memory and no globals.
    pub fn snapshot_tables(&self) -> VMTablesSnapshot {
        self.instance().snapshot_tables()
    }

    /// Restore every defined table to a previously captured [`VMTablesSnapshot`].
    pub fn reset_tables(&mut self, snapshot: &VMTablesSnapshot) -> Result<(), MemoryError> {
        self.instance_mut().restore_tables(snapshot)
    }
}
