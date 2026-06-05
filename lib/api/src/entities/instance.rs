use crate::{
    Extern, error::InstantiationError, exports::Exports, imports::Imports,
    macros::backend::gen_rt_ty, module::Module, store::AsStoreMut,
};

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
