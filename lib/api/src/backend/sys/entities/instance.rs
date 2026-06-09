//! Data types, functions and traits for `sys` runtime's `Instance` implementation.

use crate::{
    Extern, Global, error::InstantiationError, exports::Exports, imports::Imports, module::Module,
    store::{AsStoreMut, AsStoreRef},
};
use wasmer_types::{ExportIndex, GlobalIndex};
use wasmer_vm::{StoreHandle, VMInstance, VMInstanceSnapshot, VMTablesSnapshot};

use super::store::Store;

#[derive(Clone, PartialEq, Eq)]
/// A WebAssembly `instance` in the `sys` runtime.
pub struct Instance {
    _handle: StoreHandle<VMInstance>,
}

impl From<wasmer_compiler::InstantiationError> for InstantiationError {
    fn from(other: wasmer_compiler::InstantiationError) -> Self {
        match other {
            wasmer_compiler::InstantiationError::Link(e) => Self::Link(e.into()),
            wasmer_compiler::InstantiationError::Start(e) => Self::Start(e.into()),
            wasmer_compiler::InstantiationError::CpuFeature(e) => Self::CpuFeature(e),
        }
    }
}

impl Instance {
    #[allow(clippy::result_large_err)]
    pub(crate) fn new(
        store: &mut impl AsStoreMut,
        module: &Module,
        imports: &Imports,
    ) -> Result<(Self, Exports), InstantiationError> {
        let externs = imports
            .imports_for_module(module)
            .map_err(InstantiationError::Link)?;
        let mut handle = module.as_sys().instantiate(store, &externs)?;
        let exports = Self::get_exports(store, module, handle.unwrap_sys_mut());

        let instance = Self {
            _handle: StoreHandle::new(store.objects_mut().as_sys_mut(), handle.unwrap_sys()),
        };

        Ok((instance, exports))
    }

    #[allow(clippy::result_large_err)]
    pub(crate) fn new_by_index(
        store: &mut impl AsStoreMut,
        module: &Module,
        externs: &[Extern],
    ) -> Result<(Self, Exports), InstantiationError> {
        let externs = externs.to_vec();
        let mut handle = module.as_sys().instantiate(store, &externs)?;
        let exports = Self::get_exports(store, module, handle.unwrap_sys_mut());
        let instance = Self {
            _handle: StoreHandle::new(
                store.as_store_mut().objects_mut().as_sys_mut(),
                handle.unwrap_sys(),
            ),
        };

        Ok((instance, exports))
    }

    /// Return a [`Global`] handle for every DEFINED (non-imported) global in
    /// this instance, INCLUDING ones the module does not export. The caller
    /// passes the defined-global index range `[start, end)` taken from
    /// `Module::info()` (`num_imported_globals .. globals.len()`).
    ///
    /// `instance.exports` only surfaces exported globals, so a non-exported
    /// mutable global (e.g. the AssemblyScript allocator bump pointer) is
    /// otherwise invisible to an embedder and leaks state across reuses of a
    /// pooled instance. This reaches them via the VM's per-index export
    /// lookup, which does not require the global to be exported.
    pub fn defined_globals(
        &self,
        store: &mut impl AsStoreMut,
        start: usize,
        end: usize,
    ) -> Vec<Global> {
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        for idx in start..end {
            // Materialise the VM extern for this global index, then drop the
            // instance borrow before re-borrowing the store to wrap it.
            let raw = {
                let inst = self._handle.get_mut(store.objects_mut().as_sys_mut());
                inst.lookup_by_declaration(ExportIndex::Global(GlobalIndex::from_u32(idx as u32)))
            };
            if let Extern::Global(g) = Extern::from_vm_extern(store, crate::vm::VMExtern::Sys(raw)) {
                out.push(g);
            }
        }
        out
    }

    fn get_exports(
        store: &mut impl AsStoreMut,
        module: &Module,
        handle: &mut VMInstance,
    ) -> Exports {
        module
            .exports()
            .map(|export| {
                let name = export.name().to_string();
                let export = handle.lookup(&name).expect("export");
                let extern_ = Extern::from_vm_extern(store, crate::vm::VMExtern::Sys(export));
                (name, extern_)
            })
            .collect::<Exports>()
    }

    /// Capture a snapshot of this instance's mutable state (defined memories,
    /// globals, and tables). See [`wasmer_vm::VMInstance::snapshot`].
    pub(crate) fn snapshot(&self, store: &impl AsStoreRef) -> VMInstanceSnapshot {
        self._handle
            .get(store.as_store_ref().objects().as_sys())
            .snapshot()
    }

    /// Restore this instance to a previously captured snapshot.
    pub(crate) fn reset_to_snapshot(
        &self,
        store: &mut impl AsStoreMut,
        snapshot: &VMInstanceSnapshot,
    ) -> Result<(), wasmer_types::MemoryError> {
        self._handle
            .get_mut(store.as_store_mut().objects_mut().as_sys_mut())
            .reset_to_snapshot(snapshot)
    }

    /// Capture a table-only snapshot (no memory/globals). See
    /// [`wasmer_vm::VMInstance::snapshot_tables`].
    pub(crate) fn snapshot_tables(&self, store: &impl AsStoreRef) -> VMTablesSnapshot {
        self._handle
            .get(store.as_store_ref().objects().as_sys())
            .snapshot_tables()
    }

    /// Restore every defined table to a previously captured table snapshot.
    pub(crate) fn reset_tables(
        &self,
        store: &mut impl AsStoreMut,
        snapshot: &VMTablesSnapshot,
    ) -> Result<(), wasmer_types::MemoryError> {
        self._handle
            .get_mut(store.as_store_mut().objects_mut().as_sys_mut())
            .reset_tables(snapshot)
    }
}

impl crate::BackendInstance {
    /// Consume [`self`] into a [`crate::backend::sys::instance::Instance`].
    pub(crate) fn into_sys(self) -> crate::backend::sys::instance::Instance {
        match self {
            Self::Sys(s) => s,
            _ => panic!("Not a `sys` instance"),
        }
    }
}

#[cfg(test)]
mod send_test {
    use super::*;

    // Only here to statically ensure that `Instance` is `Send`.
    // Will fail to compile otherwise.
    #[allow(dead_code)]
    fn instance_is_send(inst: Instance) {
        fn is_send(t: impl Send) {
            let _ = t;
        }

        is_send(inst);
    }
}
