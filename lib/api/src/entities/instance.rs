use crate::{
    Extern, RuntimeError, error::InstantiationError, exports::Exports, imports::Imports,
    macros::backend::gen_rt_ty, module::Module, store::AsStoreMut,
};
#[cfg(feature = "sys")]
use crate::store::AsStoreRef;

/// A WebAssembly Instance is a stateful, executable
/// instance of a WebAssembly [`Module`].
///
/// Instance objects contain all the exported WebAssembly
/// functions, memories, tables and globals that allow
/// interacting with WebAssembly.
///
/// Spec: <https://webassembly.github.io/spec/core/exec/runtime.html#module-instances>
#[derive(Clone, PartialEq, Eq)]
pub struct Instance {
    pub(crate) _inner: BackendInstance,
    pub(crate) module: Module,
    /// The exports for an instance.
    pub exports: Exports,
}

impl Instance {
    /// Creates a new `Instance` from a WebAssembly [`Module`] and a
    /// set of imports using [`Imports`] or the [`imports!`] macro helper.
    ///
    /// [`imports!`]: crate::imports!
    /// [`Imports!`]: crate::Imports!
    ///
    /// ```
    /// # use wasmer::{imports, Store, Module, Global, Value, Instance};
    /// # use wasmer::FunctionEnv;
    /// # fn main() -> anyhow::Result<()> {
    /// let mut store = Store::default();
    /// let env = FunctionEnv::new(&mut store, ());
    /// let module = Module::new(&store, "(module)")?;
    /// let imports = imports!{
    ///   "host" => {
    ///     "var" => Global::new(&mut store, Value::I32(2))
    ///   }
    /// };
    /// let instance = Instance::new(&mut store, &module, &imports)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Errors
    ///
    /// The function can return [`InstantiationError`]s.
    ///
    /// Those are, as defined by the spec:
    ///  * Link errors that happen when plugging the imports into the instance
    ///  * Runtime errors that happen when running the module `start` function.
    #[allow(clippy::result_large_err)]
    pub fn new(
        store: &mut impl AsStoreMut,
        module: &Module,
        imports: &Imports,
    ) -> Result<Self, InstantiationError> {
        let (_inner, exports) = match &store.as_store_mut().inner.store {
            #[cfg(feature = "sys")]
            crate::BackendStore::Sys(_) => {
                let (i, e) = crate::backend::sys::instance::Instance::new(store, module, imports)?;
                (crate::BackendInstance::Sys(i), e)
            }
            #[cfg(feature = "v8")]
            crate::BackendStore::V8(_) => {
                let (i, e) = crate::backend::v8::instance::Instance::new(store, module, imports)?;
                (crate::BackendInstance::V8(i), e)
            }
            #[cfg(feature = "js")]
            crate::BackendStore::Js(_) => {
                let (i, e) = crate::backend::js::instance::Instance::new(store, module, imports)?;
                (crate::BackendInstance::Js(i), e)
            }
        };

        Ok(Self {
            _inner,
            module: module.clone(),
            exports,
        })
    }

    /// Creates a new `Instance` from a WebAssembly [`Module`] and a
    /// vector of imports.
    ///
    /// ## Errors
    ///
    /// The function can return [`InstantiationError`]s.
    ///
    /// Those are, as defined by the spec:
    ///  * Link errors that happen when plugging the imports into the instance
    ///  * Runtime errors that happen when running the module `start` function.
    #[allow(clippy::result_large_err)]
    pub fn new_by_index(
        store: &mut impl AsStoreMut,
        module: &Module,
        externs: &[Extern],
    ) -> Result<Self, InstantiationError> {
        let (_inner, exports) = match &store.as_store_mut().inner.store {
            #[cfg(feature = "sys")]
            crate::BackendStore::Sys(_) => {
                let (i, e) =
                    crate::backend::sys::instance::Instance::new_by_index(store, module, externs)?;
                (crate::BackendInstance::Sys(i), e)
            }
            #[cfg(feature = "v8")]
            crate::BackendStore::V8(_) => {
                let (i, e) =
                    crate::backend::v8::instance::Instance::new_by_index(store, module, externs)?;
                (crate::BackendInstance::V8(i), e)
            }
            #[cfg(feature = "js")]
            crate::BackendStore::Js(_) => {
                let (i, e) =
                    crate::backend::js::instance::Instance::new_by_index(store, module, externs)?;
                (crate::BackendInstance::Js(i), e)
            }
        };

        Ok(Self {
            _inner,
            module: module.clone(),
            exports,
        })
    }

    /// Gets the [`Module`] associated with this instance.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Return a [`crate::Global`] handle for every DEFINED (non-imported)
    /// global in this instance, INCLUDING globals the module does not
    /// export. `self.exports` only contains exported globals, so a
    /// non-exported mutable global (e.g. an allocator bump pointer that a
    /// toolchain keeps internal) is invisible there and would leak state
    /// across reuses of a pooled instance. Use this to snapshot and restore
    /// the COMPLETE mutable-global state between calls. Empty on non-`sys`
    /// backends.
    pub fn defined_globals(&self, store: &mut impl AsStoreMut) -> Vec<crate::Global> {
        let info = self.module.info();
        let start = info.num_imported_globals;
        let end = info.globals.len();
        match &self._inner {
            #[cfg(feature = "sys")]
            BackendInstance::Sys(i) => i.defined_globals(store, start, end),
            #[allow(unreachable_patterns)]
            _ => Vec::new(),
        }
    }

    /// Capture a snapshot of this instance's mutable state — its defined linear
    /// memories, globals, and tables — so the instance can be reset and reused
    /// instead of being re-instantiated.
    ///
    /// Take the snapshot once the instance has reached the baseline you want to
    /// reuse for every run (after the `start` function and any embedder
    /// initialization). Host-side state (imported memories/globals, function
    /// environments) is not captured and remains the embedder's responsibility.
    ///
    /// Only supported on the `sys` backend; other backends return an error.
    #[cfg(feature = "sys")]
    pub fn snapshot(&self, store: &impl AsStoreRef) -> Result<InstanceSnapshot, RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => Ok(InstanceSnapshot {
                inner: i.snapshot(store),
            }),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::snapshot is only supported on the sys backend",
            )),
        }
    }

    /// Capture a snapshot with an eager (memcpy) memory image, regardless of
    /// copy-on-write support. Required for [`Instance::reset_to_snapshot_bounded`]
    /// (which memcpys a prefix of the captured image). Only supported on `sys`.
    #[cfg(feature = "sys")]
    pub fn snapshot_eager(&self, store: &impl AsStoreRef) -> Result<InstanceSnapshot, RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => Ok(InstanceSnapshot {
                inner: i.snapshot_eager(store),
            }),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::snapshot_eager is only supported on the sys backend",
            )),
        }
    }

    /// Restore this instance to a previously captured [`InstanceSnapshot`],
    /// undoing every memory/global/table mutation (and any `data.drop` /
    /// `elem.drop`) performed since the snapshot was taken.
    ///
    /// Only supported on the `sys` backend; other backends return an error.
    #[cfg(feature = "sys")]
    pub fn reset_to_snapshot(
        &self,
        store: &mut impl AsStoreMut,
        snapshot: &InstanceSnapshot,
    ) -> Result<(), RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => i
                .reset_to_snapshot(store, &snapshot.inner)
                .map_err(|e| RuntimeError::new(e.to_string())),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::reset_to_snapshot is only supported on the sys backend",
            )),
        }
    }

    /// Restore from an eager [`InstanceSnapshot`], copying each memory only over
    /// its first `mem_dirty_bytes` (a plain `memcpy`, no page-table edits, so it
    /// scales across cores — unlike the copy-on-write path). The caller
    /// guarantees the tail `[mem_dirty_bytes, current_length)` is already at its
    /// reset value (e.g. held zero by a soft guard). Globals, tables, and passive
    /// segments are restored in full.
    ///
    /// The snapshot must be eager ([`Instance::snapshot_eager`]); a CoW snapshot
    /// errors. Only supported on the `sys` backend.
    #[cfg(feature = "sys")]
    pub fn reset_to_snapshot_bounded(
        &self,
        store: &mut impl AsStoreMut,
        snapshot: &InstanceSnapshot,
        mem_dirty_bytes: usize,
    ) -> Result<(), RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => i
                .reset_to_snapshot_bounded(store, &snapshot.inner, mem_dirty_bytes)
                .map_err(|e| RuntimeError::new(e.to_string())),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::reset_to_snapshot_bounded is only supported on the sys backend",
            )),
        }
    }

    /// Capture a snapshot of this instance's defined tables only — no linear
    /// memory and no globals.
    ///
    /// Use this when you already reset memory and globals yourself (e.g. a
    /// bespoke fast path) and only need correct, raw-element table restoration.
    /// Restoring copies the raw `funcref`/`externref` slots back verbatim, with
    /// no `Value` round-trip. Only supported on the `sys` backend.
    #[cfg(feature = "sys")]
    pub fn snapshot_tables(&self, store: &impl AsStoreRef) -> Result<TablesSnapshot, RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => Ok(TablesSnapshot {
                inner: i.snapshot_tables(store),
            }),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::snapshot_tables is only supported on the sys backend",
            )),
        }
    }

    /// Restore every defined table to a previously captured [`TablesSnapshot`],
    /// undoing any `table.set` / `table.grow` since the snapshot. Only supported
    /// on the `sys` backend.
    #[cfg(feature = "sys")]
    pub fn reset_tables(
        &self,
        store: &mut impl AsStoreMut,
        snapshot: &TablesSnapshot,
    ) -> Result<(), RuntimeError> {
        match &self._inner {
            BackendInstance::Sys(i) => i
                .reset_tables(store, &snapshot.inner)
                .map_err(|e| RuntimeError::new(e.to_string())),
            #[allow(unreachable_patterns)]
            _ => Err(RuntimeError::new(
                "Instance::reset_tables is only supported on the sys backend",
            )),
        }
    }
}

/// An opaque table-only snapshot produced by [`Instance::snapshot_tables`] and
/// consumed by [`Instance::reset_tables`]. Keep the originating [`Instance`]
/// alive while it is in use. Only available on the `sys` backend.
#[cfg(feature = "sys")]
pub struct TablesSnapshot {
    inner: wasmer_vm::VMTablesSnapshot,
}

#[cfg(feature = "sys")]
impl std::fmt::Debug for TablesSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

/// An opaque snapshot of an [`Instance`]'s mutable state, produced by
/// [`Instance::snapshot`] and consumed by [`Instance::reset_to_snapshot`].
///
/// Keep the originating [`Instance`] alive while the snapshot is in use. Only
/// available on the `sys` backend.
#[cfg(feature = "sys")]
pub struct InstanceSnapshot {
    inner: wasmer_vm::VMInstanceSnapshot,
}

#[cfg(feature = "sys")]
impl std::fmt::Debug for InstanceSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.inner, f)
    }
}

impl std::fmt::Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("exports", &self.exports)
            .finish()
    }
}

/// An enumeration of all the possible instances kind supported by the runtimes.
gen_rt_ty! {
    #[derive(Clone, PartialEq, Eq)]
    pub(crate) BackendInstance(entities::instance::Instance);
}
