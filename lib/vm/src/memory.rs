// This file contains code from external sources.
// Attributions: https://github.com/wasmerio/wasmer/blob/main/docs/ATTRIBUTIONS.md

//! Memory management for linear memories.
//!
//! `Memory` is to WebAssembly linear memories what `Table` is to WebAssembly tables.

use crate::threadconditions::ThreadConditions;
pub use crate::threadconditions::{NotifyLocation, WaiterError};
use crate::trap::Trap;
use crate::{
    mmap::{Mmap, MmapType},
    store::MaybeInstanceOwned,
    threadconditions::ExpectedValue,
    vmcontext::VMMemoryDefinition,
};
use more_asserts::assert_ge;
use std::cell::UnsafeCell;
use std::convert::TryInto;
use std::ptr::NonNull;
use std::slice;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use wasmer_types::{Bytes, MemoryError, MemoryStyle, MemoryType, Pages, WASM_PAGE_SIZE};

// The memory mapped area
#[derive(Debug)]
struct WasmMmap {
    // Our OS allocation of mmap'd memory.
    alloc: Mmap,
    // The current logical size in wasm pages of this linear memory.
    size: Pages,
    /// The owned memory definition used by the generated code
    vm_memory_definition: MaybeInstanceOwned<VMMemoryDefinition>,
}

/// # SAFETY: Not safe by rust standards, since guest code may do weird things
/// with its memory. However, this is still safe to send across threads as
/// far as the WASM spec is concerned.
unsafe impl Send for WasmMmap {}
/// # SAFETY: see above.
unsafe impl Sync for WasmMmap {}

impl WasmMmap {
    fn get_vm_memory_definition(&self) -> NonNull<VMMemoryDefinition> {
        self.vm_memory_definition.as_ptr()
    }

    fn size(&self) -> Pages {
        unsafe {
            let md_ptr = self.get_vm_memory_definition();
            let md = md_ptr.as_ref();
            Bytes::from(md.current_length).try_into().unwrap()
        }
    }

    fn grow(&mut self, delta: Pages, conf: VMMemoryConfig) -> Result<Pages, MemoryError> {
        // Optimization of memory.grow 0 calls.
        if delta.0 == 0 {
            return Ok(self.size);
        }

        let new_pages = self
            .size
            .checked_add(delta)
            .ok_or(MemoryError::CouldNotGrow {
                current: self.size,
                attempted_delta: delta,
            })?;
        let prev_pages = self.size;

        if let Some(maximum) = conf.maximum
            && new_pages > maximum
        {
            return Err(MemoryError::CouldNotGrow {
                current: self.size,
                attempted_delta: delta,
            });
        }

        // Wasm linear memories are never allowed to grow beyond what is
        // indexable. If the memory has no maximum, enforce the greatest
        // limit here.
        if new_pages > Pages::max_value() {
            // Linear memory size would exceed the index range.
            return Err(MemoryError::CouldNotGrow {
                current: self.size,
                attempted_delta: delta,
            });
        }

        let delta_bytes = delta.bytes().0;
        let prev_bytes = prev_pages.bytes().0;
        let new_bytes = new_pages.bytes().0;

        if new_bytes > self.alloc.len() - conf.offset_guard_size {
            // If the new size is within the declared maximum, but needs more memory than we
            // have on hand, it's a dynamic heap and it can move.
            let guard_bytes = conf.offset_guard_size;
            let request_bytes =
                new_bytes
                    .checked_add(guard_bytes)
                    .ok_or_else(|| MemoryError::CouldNotGrow {
                        current: new_pages,
                        attempted_delta: Bytes(guard_bytes).try_into().unwrap(),
                    })?;

            let mut new_mmap =
                Mmap::accessible_reserved(new_bytes, request_bytes, None, MmapType::Private)
                    .map_err(MemoryError::Region)?;

            let copy_len = self.alloc.len() - conf.offset_guard_size;
            new_mmap.as_mut_slice()[..copy_len].copy_from_slice(&self.alloc.as_slice()[..copy_len]);

            self.alloc = new_mmap;
        } else if delta_bytes > 0 {
            // Make the newly allocated pages accessible.
            self.alloc
                .make_accessible(prev_bytes, delta_bytes)
                .map_err(MemoryError::Region)?;
        }

        self.size = new_pages;

        // update memory definition
        unsafe {
            let mut md_ptr = self.vm_memory_definition.as_ptr();
            let md = md_ptr.as_mut();
            md.current_length = new_pages.bytes().0;
            md.base = self.alloc.as_mut_ptr() as _;
        }

        Ok(prev_pages)
    }

    /// Grows the memory to at least a minimum size. If the memory is already big enough
    /// for the min size then this function does nothing
    fn grow_at_least(&mut self, min_size: u64, conf: VMMemoryConfig) -> Result<(), MemoryError> {
        let cur_size = self.size.bytes().0 as u64;
        if cur_size < min_size {
            let growth = min_size - cur_size;
            let growth_pages = ((growth - 1) / WASM_PAGE_SIZE as u64) + 1;
            self.grow(Pages(growth_pages as u32), conf)?;
        }

        Ok(())
    }

    /// Resets the memory down to a zero size
    fn reset(&mut self) -> Result<(), MemoryError> {
        self.size.0 = 0;
        // update memory definition
        unsafe {
            let mut md_ptr = self.vm_memory_definition.as_ptr();
            let md = md_ptr.as_mut();
            md.current_length = 0;
        }
        Ok(())
    }

    /// Restore the memory image and logical size to a captured snapshot.
    ///
    /// Grows the backing allocation if the snapshot is larger than the current
    /// logical size, copies `image` into `[0, image.len())`, zeroes any pages
    /// above the snapshot (so a later grow sees zeros), then shrinks the logical
    /// size back down to the snapshot.
    fn restore_image(&mut self, image: &[u8], conf: VMMemoryConfig) -> Result<(), MemoryError> {
        let target_bytes = image.len();
        let target_pages: Pages = Bytes::from(target_bytes)
            .try_into()
            .map_err(|_| MemoryError::Generic("snapshot image size overflows pages".into()))?;

        // Grow back up if the snapshot is larger than the current logical size.
        if target_pages > self.size {
            let delta = target_pages - self.size;
            self.grow(delta, conf)?;
        }

        // `[0, self.size)` is now accessible and `self.size >= target_pages`.
        let (base, live_bytes) = unsafe {
            let md = self.vm_memory_definition.as_ptr();
            let md = md.as_ref();
            (md.base, md.current_length)
        };
        unsafe {
            if live_bytes > target_bytes {
                // Zero pages above the snapshot so a future `memory.grow` exposes
                // zero-initialized memory (Wasm spec) and no prior-run data leaks.
                std::ptr::write_bytes(base.add(target_bytes), 0u8, live_bytes - target_bytes);
            }
            slice::from_raw_parts_mut(base, target_bytes).copy_from_slice(image);
        }

        // Shrink the logical view back down to the snapshot size.
        self.size = target_pages;
        unsafe {
            let mut md = self.vm_memory_definition.as_ptr();
            md.as_mut().current_length = target_bytes;
        }
        Ok(())
    }

    /// Copies the memory
    /// (in this case it performs a copy-on-write to save memory)
    pub fn copy(&self) -> Result<Self, MemoryError> {
        let mem_length = self.size.bytes().0;
        let mut alloc = self
            .alloc
            .copy(Some(mem_length))
            .map_err(MemoryError::Generic)?;
        let base_ptr = alloc.as_mut_ptr();
        Ok(Self {
            vm_memory_definition: MaybeInstanceOwned::Host(Box::new(UnsafeCell::new(
                VMMemoryDefinition {
                    base: base_ptr,
                    current_length: mem_length,
                },
            ))),
            alloc,
            size: self.size,
        })
    }
}

/// A linear memory instance.
#[derive(Debug, Clone)]
struct VMMemoryConfig {
    // The optional maximum size in wasm pages of this linear memory.
    maximum: Option<Pages>,
    /// The WebAssembly linear memory description.
    memory: MemoryType,
    /// Our chosen implementation style.
    style: MemoryStyle,
    // Size in bytes of extra guard pages after the end to optimize loads and stores with
    // constant offsets.
    offset_guard_size: usize,
}

impl VMMemoryConfig {
    fn ty(&self, minimum: Pages) -> MemoryType {
        let mut out = self.memory;
        out.minimum = minimum;

        out
    }

    fn style(&self) -> MemoryStyle {
        self.style
    }
}

/// A linear memory instance.
#[derive(Debug)]
pub struct VMOwnedMemory {
    // The underlying allocation.
    mmap: WasmMmap,
    // Configuration of this memory
    config: VMMemoryConfig,
}

unsafe impl Send for VMOwnedMemory {}
unsafe impl Sync for VMOwnedMemory {}

impl VMOwnedMemory {
    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with owned metadata: this can be used to create a memory
    /// that will be imported into Wasm modules.
    pub fn new(memory: &MemoryType, style: &MemoryStyle) -> Result<Self, MemoryError> {
        unsafe { Self::new_internal(memory, style, None, None, MmapType::Private) }
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages
    /// that is backed by a memory file. When set to private the file will be remaining in memory and
    /// never flush to disk, when set to shared the memory will be flushed to disk.
    ///
    /// This creates a `Memory` with owned metadata: this can be used to create a memory
    /// that will be imported into Wasm modules.
    pub fn new_with_file(
        memory: &MemoryType,
        style: &MemoryStyle,
        backing_file: std::path::PathBuf,
        memory_type: MmapType,
    ) -> Result<Self, MemoryError> {
        unsafe { Self::new_internal(memory, style, None, Some(backing_file), memory_type) }
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with metadata owned by a VM, pointed to by
    /// `vm_memory_location`: this can be used to create a local memory.
    ///
    /// # Safety
    /// - `vm_memory_location` must point to a valid location in VM memory.
    pub unsafe fn from_definition(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: NonNull<VMMemoryDefinition>,
    ) -> Result<Self, MemoryError> {
        unsafe {
            Self::new_internal(
                memory,
                style,
                Some(vm_memory_location),
                None,
                MmapType::Private,
            )
        }
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages
    /// that is backed by a file. When set to private the file will be remaining in memory and
    /// never flush to disk, when set to shared the memory will be flushed to disk.
    ///
    /// This creates a `Memory` with metadata owned by a VM, pointed to by
    /// `vm_memory_location`: this can be used to create a local memory.
    ///
    /// # Safety
    /// - `vm_memory_location` must point to a valid location in VM memory.
    pub unsafe fn from_definition_with_file(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: NonNull<VMMemoryDefinition>,
        backing_file: Option<std::path::PathBuf>,
        memory_type: MmapType,
    ) -> Result<Self, MemoryError> {
        unsafe {
            Self::new_internal(
                memory,
                style,
                Some(vm_memory_location),
                backing_file,
                memory_type,
            )
        }
    }

    /// Build a `Memory` with either self-owned or VM owned metadata.
    unsafe fn new_internal(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: Option<NonNull<VMMemoryDefinition>>,
        backing_file: Option<std::path::PathBuf>,
        memory_type: MmapType,
    ) -> Result<Self, MemoryError> {
        unsafe {
            if memory.minimum > Pages::max_value() {
                return Err(MemoryError::MinimumMemoryTooLarge {
                    min_requested: memory.minimum,
                    max_allowed: Pages::max_value(),
                });
            }
            // `maximum` cannot be set to more than `65536` pages.
            if let Some(max) = memory.maximum {
                if max > Pages::max_value() {
                    return Err(MemoryError::MaximumMemoryTooLarge {
                        max_requested: max,
                        max_allowed: Pages::max_value(),
                    });
                }
                if max < memory.minimum {
                    return Err(MemoryError::InvalidMemory {
                        reason: format!(
                            "the maximum ({} pages) is less than the minimum ({} pages)",
                            max.0, memory.minimum.0
                        ),
                    });
                }
            }

            let offset_guard_bytes = style.offset_guard_size() as usize;

            let minimum_pages = match style {
                MemoryStyle::Dynamic { .. } => memory.minimum,
                MemoryStyle::Static { bound, .. } => {
                    assert_ge!(*bound, memory.minimum);
                    *bound
                }
            };
            let minimum_bytes = minimum_pages.bytes().0;
            let request_bytes = minimum_bytes.checked_add(offset_guard_bytes).unwrap();
            let mapped_pages = memory.minimum;
            let mapped_bytes = mapped_pages.bytes();

            let mut alloc =
                Mmap::accessible_reserved(mapped_bytes.0, request_bytes, backing_file, memory_type)
                    .map_err(MemoryError::Region)?;

            let base_ptr = alloc.as_mut_ptr();
            let mem_length = memory
                .minimum
                .bytes()
                .0
                .max(alloc.as_slice_accessible().len());
            let mmap = WasmMmap {
                vm_memory_definition: if let Some(mem_loc) = vm_memory_location {
                    {
                        let mut ptr = mem_loc;
                        let md = ptr.as_mut();
                        md.base = base_ptr;
                        md.current_length = mem_length;
                    }
                    MaybeInstanceOwned::Instance(mem_loc)
                } else {
                    MaybeInstanceOwned::Host(Box::new(UnsafeCell::new(VMMemoryDefinition {
                        base: base_ptr,
                        current_length: mem_length,
                    })))
                },
                alloc,
                size: Bytes::from(mem_length).try_into().unwrap(),
            };

            Ok(Self {
                mmap,
                config: VMMemoryConfig {
                    maximum: memory.maximum,
                    offset_guard_size: offset_guard_bytes,
                    memory: *memory,
                    style: *style,
                },
            })
        }
    }

    /// Converts this owned memory into shared memory
    pub fn to_shared(self) -> VMSharedMemory {
        VMSharedMemory {
            mmap: Arc::new(RwLock::new(self.mmap)),
            config: self.config,
            conditions: ThreadConditions::new(),
        }
    }

    /// Copies this memory to a new memory
    pub fn copy(&self) -> Result<Self, MemoryError> {
        Ok(Self {
            mmap: self.mmap.copy()?,
            config: self.config.clone(),
        })
    }
}

// TODO: why doesn't this support wait/notify? wait should block indefinitely if you ask me
impl LinearMemory for VMOwnedMemory {
    /// Returns the type for this memory.
    fn ty(&self) -> MemoryType {
        let minimum = self.mmap.size();
        self.config.ty(minimum)
    }

    /// Returns the size of the memory in pages
    fn size(&self) -> Pages {
        self.mmap.size()
    }

    /// Returns the memory style for this memory.
    fn style(&self) -> MemoryStyle {
        self.config.style()
    }

    /// Grow memory by the specified amount of wasm pages.
    ///
    /// Returns `None` if memory can't be grown by the specified amount
    /// of wasm pages.
    fn grow(&mut self, delta: Pages) -> Result<Pages, MemoryError> {
        self.mmap.grow(delta, self.config.clone())
    }

    /// Grows the memory to at least a minimum size. If the memory is already big enough
    /// for the min size then this function does nothing
    fn grow_at_least(&mut self, min_size: u64) -> Result<(), MemoryError> {
        self.mmap.grow_at_least(min_size, self.config.clone())
    }

    /// Resets the memory down to a zero size
    fn reset(&mut self) -> Result<(), MemoryError> {
        self.mmap.reset()?;
        Ok(())
    }

    /// Restore the memory contents and size to a captured snapshot image.
    fn restore_image(&mut self, image: &[u8]) -> Result<(), MemoryError> {
        self.mmap.restore_image(image, self.config.clone())
    }

    /// Copy-on-write snapshots are supported for static-style memories on Linux,
    /// whose base address never moves on `memory.grow`.
    fn supports_cow_snapshot(&self) -> bool {
        cfg!(target_os = "linux") && matches!(self.config.style(), MemoryStyle::Static { .. })
    }

    fn snapshot_cow(&self) -> Result<CowBacking, MemoryError> {
        #[cfg(target_os = "linux")]
        {
            self.mmap.snapshot_cow()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Err(MemoryError::UnsupportedOperation {
                message: "snapshot_cow() is only supported on Linux".to_string(),
            })
        }
    }

    fn restore_cow(&mut self, cow: &CowBacking) -> Result<(), MemoryError> {
        #[cfg(target_os = "linux")]
        {
            self.mmap.restore_cow(cow)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = cow;
            Err(MemoryError::UnsupportedOperation {
                message: "restore_cow() is only supported on Linux".to_string(),
            })
        }
    }

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm code.
    fn vmmemory(&self) -> NonNull<VMMemoryDefinition> {
        self.mmap.vm_memory_definition.as_ptr()
    }

    /// Owned memory can not be cloned (this will always return None)
    fn try_clone(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        Err(MemoryError::MemoryNotShared)
    }

    /// Copies this memory to a new memory
    fn copy(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        let forked = Self::copy(self)?;
        Ok(Box::new(forked))
    }

    /// Return a concrete shared memory handle for detached API sharing.
    fn as_shared(&self) -> Result<VMSharedMemory, MemoryError> {
        Err(MemoryError::MemoryNotShared)
    }
}

/// A shared linear memory instance.
#[derive(Debug, Clone)]
pub struct VMSharedMemory {
    // The underlying allocation.
    mmap: Arc<RwLock<WasmMmap>>,
    // Configuration of this memory
    config: VMMemoryConfig,
    // waiters list for this memory
    conditions: ThreadConditions,
}

impl VMSharedMemory {
    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with owned metadata: this can be used to create a memory
    /// that will be imported into Wasm modules.
    pub fn new(memory: &MemoryType, style: &MemoryStyle) -> Result<Self, MemoryError> {
        Ok(VMOwnedMemory::new(memory, style)?.to_shared())
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages
    /// that is backed by a file. When set to private the file will be remaining in memory and
    /// never flush to disk, when set to shared the memory will be flushed to disk.
    ///
    /// This creates a `Memory` with owned metadata: this can be used to create a memory
    /// that will be imported into Wasm modules.
    pub fn new_with_file(
        memory: &MemoryType,
        style: &MemoryStyle,
        backing_file: std::path::PathBuf,
        memory_type: MmapType,
    ) -> Result<Self, MemoryError> {
        Ok(VMOwnedMemory::new_with_file(memory, style, backing_file, memory_type)?.to_shared())
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with metadata owned by a VM, pointed to by
    /// `vm_memory_location`: this can be used to create a local memory.
    ///
    /// # Safety
    /// - `vm_memory_location` must point to a valid location in VM memory.
    pub unsafe fn from_definition(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: NonNull<VMMemoryDefinition>,
    ) -> Result<Self, MemoryError> {
        unsafe {
            Ok(VMOwnedMemory::from_definition(memory, style, vm_memory_location)?.to_shared())
        }
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages
    /// that is backed by a file. When set to private the file will be remaining in memory and
    /// never flush to disk, when set to shared the memory will be flushed to disk.
    ///
    /// This creates a `Memory` with metadata owned by a VM, pointed to by
    /// `vm_memory_location`: this can be used to create a local memory.
    ///
    /// # Safety
    /// - `vm_memory_location` must point to a valid location in VM memory.
    pub unsafe fn from_definition_with_file(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: NonNull<VMMemoryDefinition>,
        backing_file: Option<std::path::PathBuf>,
        memory_type: MmapType,
    ) -> Result<Self, MemoryError> {
        unsafe {
            Ok(VMOwnedMemory::from_definition_with_file(
                memory,
                style,
                vm_memory_location,
                backing_file,
                memory_type,
            )?
            .to_shared())
        }
    }

    /// Copies this memory to a new memory
    pub fn copy(&self) -> Result<Self, MemoryError> {
        let guard = self.mmap.read().unwrap();
        Ok(Self {
            mmap: Arc::new(RwLock::new(guard.copy()?)),
            config: self.config.clone(),
            conditions: ThreadConditions::new(),
        })
    }
}

impl LinearMemory for VMSharedMemory {
    /// Returns the type for this memory.
    fn ty(&self) -> MemoryType {
        let minimum = {
            let guard = self.mmap.read().unwrap();
            guard.size()
        };
        self.config.ty(minimum)
    }

    /// Returns the size of the memory in pages
    fn size(&self) -> Pages {
        let guard = self.mmap.read().unwrap();
        guard.size()
    }

    /// Returns the memory style for this memory.
    fn style(&self) -> MemoryStyle {
        self.config.style()
    }

    /// Grow memory by the specified amount of wasm pages.
    ///
    /// Returns `None` if memory can't be grown by the specified amount
    /// of wasm pages.
    fn grow(&mut self, delta: Pages) -> Result<Pages, MemoryError> {
        let mut guard = self.mmap.write().unwrap();
        guard.grow(delta, self.config.clone())
    }

    /// Grows the memory to at least a minimum size. If the memory is already big enough
    /// for the min size then this function does nothing
    fn grow_at_least(&mut self, min_size: u64) -> Result<(), MemoryError> {
        let mut guard = self.mmap.write().unwrap();
        guard.grow_at_least(min_size, self.config.clone())
    }

    /// Resets the memory down to a zero size
    fn reset(&mut self) -> Result<(), MemoryError> {
        let mut guard = self.mmap.write().unwrap();
        guard.reset()?;
        Ok(())
    }

    /// Restore the memory contents and size to a captured snapshot image.
    fn restore_image(&mut self, image: &[u8]) -> Result<(), MemoryError> {
        let mut guard = self.mmap.write().unwrap();
        guard.restore_image(image, self.config.clone())
    }

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm code.
    fn vmmemory(&self) -> NonNull<VMMemoryDefinition> {
        let guard = self.mmap.read().unwrap();
        guard.vm_memory_definition.as_ptr()
    }

    /// Shared memory can always be cloned
    fn try_clone(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        Ok(Box::new(self.clone()))
    }

    /// Copies this memory to a new memory
    fn copy(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        let forked = Self::copy(self)?;
        Ok(Box::new(forked))
    }

    /// Return a concrete shared memory handle for detached API sharing.
    fn as_shared(&self) -> Result<VMSharedMemory, MemoryError> {
        Ok(self.clone())
    }

    // Add current thread to waiter list
    unsafe fn do_wait(
        &mut self,
        dst: u32,
        expected: ExpectedValue,
        timeout: Option<Duration>,
    ) -> Result<u32, WaiterError> {
        let dst = NotifyLocation {
            address: dst,
            memory_base: self.mmap.read().unwrap().alloc.as_ptr() as *mut _,
        };
        unsafe { self.conditions.do_wait(dst, expected, timeout) }
    }

    /// Notify waiters from the wait list. Return the number of waiters notified
    fn do_notify(&mut self, dst: u32, count: u32) -> u32 {
        self.conditions.do_notify(dst, count)
    }

    fn thread_conditions(&self) -> Option<&ThreadConditions> {
        Some(&self.conditions)
    }
}

impl From<VMOwnedMemory> for VMMemory {
    fn from(mem: VMOwnedMemory) -> Self {
        Self(Box::new(mem))
    }
}

impl From<VMSharedMemory> for VMMemory {
    fn from(mem: VMSharedMemory) -> Self {
        Self(Box::new(mem))
    }
}

/// Represents linear memory that can be either owned or shared
#[derive(Debug)]
pub struct VMMemory(pub Box<dyn LinearMemory + Send + Sync + 'static>);

impl From<Box<dyn LinearMemory + Send + Sync + 'static>> for VMMemory {
    fn from(mem: Box<dyn LinearMemory + Send + Sync + 'static>) -> Self {
        Self(mem)
    }
}

impl LinearMemory for VMMemory {
    /// Returns the type for this memory.
    fn ty(&self) -> MemoryType {
        self.0.ty()
    }

    /// Returns the size of the memory in pages
    fn size(&self) -> Pages {
        self.0.size()
    }

    /// Grow memory by the specified amount of wasm pages.
    ///
    /// Returns `None` if memory can't be grown by the specified amount
    /// of wasm pages.
    fn grow(&mut self, delta: Pages) -> Result<Pages, MemoryError> {
        self.0.grow(delta)
    }

    /// Grows the memory to at least a minimum size. If the memory is already big enough
    /// for the min size then this function does nothing
    fn grow_at_least(&mut self, min_size: u64) -> Result<(), MemoryError> {
        self.0.grow_at_least(min_size)
    }

    /// Resets the memory down to a zero size
    fn reset(&mut self) -> Result<(), MemoryError> {
        self.0.reset()?;
        Ok(())
    }

    /// Restore the memory contents and size to a captured snapshot image.
    fn restore_image(&mut self, image: &[u8]) -> Result<(), MemoryError> {
        self.0.restore_image(image)
    }

    fn supports_cow_snapshot(&self) -> bool {
        self.0.supports_cow_snapshot()
    }

    fn snapshot_cow(&self) -> Result<CowBacking, MemoryError> {
        self.0.snapshot_cow()
    }

    fn restore_cow(&mut self, cow: &CowBacking) -> Result<(), MemoryError> {
        self.0.restore_cow(cow)
    }

    /// Returns the memory style for this memory.
    fn style(&self) -> MemoryStyle {
        self.0.style()
    }

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm code.
    fn vmmemory(&self) -> NonNull<VMMemoryDefinition> {
        self.0.vmmemory()
    }

    /// Attempts to clone this memory (if its cloneable)
    fn try_clone(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        self.0.try_clone()
    }

    /// Initialize memory with data
    unsafe fn initialize_with_data(&self, start: usize, data: &[u8]) -> Result<(), Trap> {
        unsafe { self.0.initialize_with_data(start, data) }
    }

    /// Copies this memory to a new memory
    fn copy(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        self.0.copy()
    }

    fn as_shared(&self) -> Result<VMSharedMemory, MemoryError> {
        self.0.as_shared()
    }

    // Add current thread to waiter list
    unsafe fn do_wait(
        &mut self,
        dst: u32,
        expected: ExpectedValue,
        timeout: Option<Duration>,
    ) -> Result<u32, WaiterError> {
        unsafe { self.0.do_wait(dst, expected, timeout) }
    }

    /// Notify waiters from the wait list. Return the number of waiters notified
    fn do_notify(&mut self, dst: u32, count: u32) -> u32 {
        self.0.do_notify(dst, count)
    }

    fn thread_conditions(&self) -> Option<&ThreadConditions> {
        self.0.thread_conditions()
    }
}

impl VMMemory {
    /// Creates a new linear memory instance of the correct type with specified
    /// minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with owned metadata: this can be used to create a memory
    /// that will be imported into Wasm modules.
    pub fn new(memory: &MemoryType, style: &MemoryStyle) -> Result<Self, MemoryError> {
        Ok(if memory.shared {
            Self(Box::new(VMSharedMemory::new(memory, style)?))
        } else {
            Self(Box::new(VMOwnedMemory::new(memory, style)?))
        })
    }

    /// Returns the number of pages in the allocated memory block
    pub fn get_runtime_size(&self) -> u32 {
        self.0.size().0
    }

    /// Create a new linear memory instance with specified minimum and maximum number of wasm pages.
    ///
    /// This creates a `Memory` with metadata owned by a VM, pointed to by
    /// `vm_memory_location`: this can be used to create a local memory.
    ///
    /// # Safety
    /// - `vm_memory_location` must point to a valid location in VM memory.
    pub unsafe fn from_definition(
        memory: &MemoryType,
        style: &MemoryStyle,
        vm_memory_location: NonNull<VMMemoryDefinition>,
    ) -> Result<Self, MemoryError> {
        unsafe {
            Ok(if memory.shared {
                Self(Box::new(VMSharedMemory::from_definition(
                    memory,
                    style,
                    vm_memory_location,
                )?))
            } else {
                Self(Box::new(VMOwnedMemory::from_definition(
                    memory,
                    style,
                    vm_memory_location,
                )?))
            })
        }
    }

    /// Creates VMMemory from a custom implementation - the following into implementations
    /// are natively supported
    /// - VMOwnedMemory -> VMMemory
    /// - Box<dyn LinearMemory + 'static> -> VMMemory
    pub fn from_custom<IntoVMMemory>(memory: IntoVMMemory) -> Self
    where
        IntoVMMemory: Into<Self>,
    {
        memory.into()
    }

    /// Copies this memory to a new memory
    pub fn copy(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError> {
        LinearMemory::copy(self)
    }

    /// Attempts to clone this memory handle.
    pub fn try_clone(&self) -> Result<Self, MemoryError> {
        LinearMemory::try_clone(self).map(Self)
    }
}

#[doc(hidden)]
/// Default implementation to initialize memory with data
pub unsafe fn initialize_memory_with_data(
    memory: &VMMemoryDefinition,
    start: usize,
    data: &[u8],
) -> Result<(), Trap> {
    unsafe {
        let mem_slice = slice::from_raw_parts_mut(memory.base, memory.current_length);
        let end = start + data.len();
        let to_init = &mut mem_slice[start..end];
        to_init.copy_from_slice(data);

        Ok(())
    }
}

/// A pristine copy-on-write snapshot of a linear memory's contents.
///
/// On Linux this is a `memfd` holding the snapshot image. Restoring re-maps the
/// live memory `MAP_PRIVATE | MAP_FIXED` over the memfd, which atomically
/// discards every dirtied (private) page and re-exposes the pristine image in a
/// single syscall — no userspace copy, and the cost is independent of how much
/// the run wrote. See [`LinearMemory::snapshot_cow`] / [`LinearMemory::restore_cow`].
#[derive(Debug)]
#[allow(dead_code)] // `snap_bytes`/`pages`/`base` are unused on non-Linux targets.
pub struct CowBacking {
    /// `memfd` holding the pristine snapshot image (Linux only).
    #[cfg(target_os = "linux")]
    fd: std::os::fd::OwnedFd,
    /// Size of the snapshot image in bytes (a whole number of wasm pages).
    snap_bytes: usize,
    /// Logical size of the memory at snapshot time.
    pages: Pages,
    /// Base address the snapshot was captured at; CoW restore requires the live
    /// memory's base to be unchanged (guaranteed for static-style memories,
    /// whose reservation never moves on `memory.grow`).
    base: usize,
}

#[cfg(target_os = "linux")]
impl WasmMmap {
    /// Capture the current contents into a fresh `memfd`. Does not touch the
    /// live mapping.
    fn snapshot_cow(&self) -> Result<CowBacking, MemoryError> {
        use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

        let (base, snap_bytes) = unsafe {
            let md = self.vm_memory_definition.as_ptr();
            let md = md.as_ref();
            (md.base, md.current_length)
        };

        let fd = unsafe {
            let raw = libc::memfd_create(c"wasmer-instance-snapshot".as_ptr(), libc::MFD_CLOEXEC);
            if raw < 0 {
                return Err(MemoryError::Generic(format!(
                    "memfd_create failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            OwnedFd::from_raw_fd(raw)
        };

        if snap_bytes > 0 {
            if unsafe { libc::ftruncate(fd.as_raw_fd(), snap_bytes as libc::off_t) } != 0 {
                return Err(MemoryError::Generic(format!(
                    "ftruncate failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            // Copy the live bytes into the memfd via a temporary shared mapping.
            unsafe {
                let dst = libc::mmap(
                    std::ptr::null_mut(),
                    snap_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    fd.as_raw_fd(),
                    0,
                );
                if dst == libc::MAP_FAILED {
                    return Err(MemoryError::Generic(format!(
                        "mmap(memfd) failed: {}",
                        std::io::Error::last_os_error()
                    )));
                }
                std::ptr::copy_nonoverlapping(base, dst.cast::<u8>(), snap_bytes);
                libc::munmap(dst, snap_bytes);
            }
        }

        Ok(CowBacking {
            fd,
            snap_bytes,
            pages: self.size,
            base: base as usize,
        })
    }

    /// Restore from a CoW backing: re-map the snapshot region `MAP_PRIVATE |
    /// MAP_FIXED` over the memfd (discarding all dirtied pages), drop any pages
    /// grown above the snapshot, then shrink the logical size back.
    fn restore_cow(&mut self, cow: &CowBacking) -> Result<(), MemoryError> {
        use std::os::fd::AsRawFd;

        let (base, cur_bytes) = unsafe {
            let md = self.vm_memory_definition.as_ptr();
            let md = md.as_ref();
            (md.base, md.current_length)
        };

        // CoW restore is only valid while the base is stable (static memories).
        // A moved base means a dynamic realloc happened; refuse rather than map
        // over the wrong address.
        if base as usize != cow.base {
            return Err(MemoryError::Generic(
                "memory base moved since snapshot; cow restore is invalid".into(),
            ));
        }

        if cow.snap_bytes > 0 {
            let res = unsafe {
                libc::mmap(
                    base.cast::<libc::c_void>(),
                    cow.snap_bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_FIXED,
                    cow.fd.as_raw_fd(),
                    0,
                )
            };
            if res == libc::MAP_FAILED {
                return Err(MemoryError::Generic(format!(
                    "MAP_FIXED cow remap failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        // Drop any pages the run grew above the snapshot, so a later grow sees
        // zero-initialized memory (Wasm spec) and no prior-run data leaks.
        if cur_bytes > cow.snap_bytes {
            let rc = unsafe {
                libc::madvise(
                    base.add(cow.snap_bytes).cast::<libc::c_void>(),
                    cur_bytes - cow.snap_bytes,
                    libc::MADV_DONTNEED,
                )
            };
            if rc != 0 {
                return Err(MemoryError::Generic(format!(
                    "madvise(MADV_DONTNEED) failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        // Shrink the logical view back to the snapshot size.
        self.size = cow.pages;
        unsafe {
            let mut md = self.vm_memory_definition.as_ptr();
            md.as_mut().current_length = cow.snap_bytes;
        }
        Ok(())
    }
}

/// Extents at or below this zero with TEMPORAL stores (plain `memset`);
/// only larger extents use the non-temporal path.
///
/// Why a threshold: for a per-request instance reset, the zeroed span is
/// the instance's OWN working set — the very bytes the next request
/// rewrites. Temporal zeroing keeps those lines in L2 where the next
/// request hits them warm and the DRAM bus sees only occasional
/// write-backs. Non-temporal stores force EVERY byte to DRAM EVERY
/// reset: measured at 707k req/s with an ~80 KiB dirty extent, that was
/// ~55 GB/s of mandatory DRAM writes (5x the read traffic) — the
/// memory bus, not the CPU, capped the whole server. NT still wins for
/// huge extents (multi-MiB grown heaps) that would flush a core's
/// entire L2 through the cache hierarchy.
///
/// 512 KiB = half a typical per-core L2: a hot extent up to this size
/// stays resident between requests.
const NT_ZERO_THRESHOLD: usize = 512 * 1024;

/// Zero `[ptr, ptr + len)` — temporal `memset` up to
/// [`NT_ZERO_THRESHOLD`], cache-bypassing non-temporal stores above it
/// (x86_64; a normal `memset` elsewhere), ending with an `sfence` so the
/// weakly-ordered streaming stores are globally visible before the
/// memory is next read.
///
/// # Safety
/// `[ptr, ptr + len)` must be one valid, writable allocation.
#[cfg(target_arch = "x86_64")]
unsafe fn nt_zero(mut ptr: *mut u8, mut len: usize) {
    if len <= NT_ZERO_THRESHOLD {
        // Hot-extent fast path: stays in cache, off the DRAM bus.
        unsafe { core::ptr::write_bytes(ptr, 0, len) };
        return;
    }
    use core::arch::x86_64::{_mm_setzero_si128, _mm_sfence, _mm_stream_si128};
    unsafe {
        // Scalar until 16-byte aligned: `movntdq` requires an aligned address.
        while len > 0 && (ptr as usize & 0xf) != 0 {
            ptr.write(0);
            ptr = ptr.add(1);
            len -= 1;
        }
        let zero = _mm_setzero_si128();
        while len >= 16 {
            _mm_stream_si128(ptr.cast(), zero);
            ptr = ptr.add(16);
            len -= 16;
        }
        while len > 0 {
            ptr.write(0);
            ptr = ptr.add(1);
            len -= 1;
        }
        // Order the non-temporal stores before any subsequent load of this range.
        _mm_sfence();
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn nt_zero(ptr: *mut u8, len: usize) {
    unsafe { core::ptr::write_bytes(ptr, 0, len) }
}

/// Represents memory that is used by the WebAssembly module
pub trait LinearMemory
where
    Self: std::fmt::Debug + Send,
{
    /// Returns the type for this memory.
    fn ty(&self) -> MemoryType;

    /// Returns the size of the memory in pages
    fn size(&self) -> Pages;

    /// Returns the memory style for this memory.
    fn style(&self) -> MemoryStyle;

    /// Grow memory by the specified amount of wasm pages.
    ///
    /// Returns `None` if memory can't be grown by the specified amount
    /// of wasm pages.
    fn grow(&mut self, delta: Pages) -> Result<Pages, MemoryError>;

    /// Grows the memory to at least a minimum size. If the memory is already big enough
    /// for the min size then this function does nothing
    fn grow_at_least(&mut self, _min_size: u64) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation {
            message: "grow_at_least() is not supported".to_string(),
        })
    }

    /// Resets the memory back to zero length
    fn reset(&mut self) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation {
            message: "reset() is not supported".to_string(),
        })
    }

    /// Capture the current contents of this memory as a byte image.
    ///
    /// The image is exactly `current_length` bytes (a whole number of wasm
    /// pages). Pair with [`LinearMemory::restore_image`] to snapshot and later
    /// restore a warm instance's linear memory.
    fn snapshot_image(&self) -> Vec<u8> {
        unsafe {
            let def = self.vmmemory().as_ref();
            slice::from_raw_parts(def.base, def.current_length).to_vec()
        }
    }

    /// Restore this memory's contents and logical size to a previously captured
    /// [`snapshot_image`](LinearMemory::snapshot_image).
    ///
    /// After the call the memory is byte-for-byte equal to `image` and its
    /// `current_length` equals `image.len()`. If the memory had grown past the
    /// snapshot size, the excess pages are zeroed so a later `memory.grow`
    /// observes zero-initialized memory, as the spec requires.
    fn restore_image(&mut self, _image: &[u8]) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation {
            message: "restore_image() is not supported".to_string(),
        })
    }

    /// Restore this memory's first `zero_to` bytes from a captured `image`,
    /// leaving the tail and the logical size untouched.
    ///
    /// This is the embedder-bounded fast path: the caller has learned the dirty
    /// extent (e.g. via a `mincore`/`mprotect` guard) and guarantees the tail
    /// `[zero_to, current_length)` is already at its reset value, so only the
    /// touched prefix needs restoring. No page-table edits and no TLB shootdown,
    /// so it scales across cores, unlike the CoW path.
    ///
    /// The restore is split at `copy_prefix` (the image's highest non-zero byte,
    /// supplied by the snapshot): `[0, copy_prefix)` is copied from `image` with
    /// an ordinary temporal `memcpy`, and `[copy_prefix, zero_to)` — which the
    /// image is all-zero over — is zeroed with cache-bypassing non-temporal
    /// stores. For a tenant whose static data is a small low prefix and whose
    /// dirtied heap is large, this keeps the bulk zero-fill out of the CPU cache
    /// and preserves the embedder's hot working set under concurrent load.
    fn restore_image_bounded(
        &mut self,
        image: &[u8],
        copy_prefix: usize,
        zero_to: usize,
    ) -> Result<(), MemoryError> {
        unsafe {
            let def = self.vmmemory().as_ref();
            let zero_to = zero_to.min(image.len()).min(def.current_length);
            let copy = copy_prefix.min(zero_to);
            if copy > 0 {
                slice::from_raw_parts_mut(def.base, copy).copy_from_slice(&image[..copy]);
            }
            // The snapshot sets `copy_prefix` to the image's highest non-zero
            // byte, so `image[copy..zero_to]` is all zero and writing zeros there
            // is byte-identical to copying the image — just cache-bypassing.
            debug_assert!(
                image
                    .get(copy..zero_to.min(image.len()))
                    .is_none_or(|s| s.iter().all(|&b| b == 0)),
                "restore_image_bounded: image must be zero in [copy_prefix, zero_to)"
            );
            let zlen = zero_to - copy;
            if zlen > 0 {
                nt_zero(def.base.add(copy), zlen);
            }
        }
        Ok(())
    }

    /// Whether this memory supports low-latency copy-on-write snapshots
    /// ([`snapshot_cow`](LinearMemory::snapshot_cow)). When `false`, callers
    /// fall back to the eager [`snapshot_image`](LinearMemory::snapshot_image)
    /// path.
    fn supports_cow_snapshot(&self) -> bool {
        false
    }

    /// Capture the current contents as a copy-on-write [`CowBacking`] for
    /// O(dirty-pages) restore. Only call when
    /// [`supports_cow_snapshot`](LinearMemory::supports_cow_snapshot) is `true`.
    fn snapshot_cow(&self) -> Result<CowBacking, MemoryError> {
        Err(MemoryError::UnsupportedOperation {
            message: "snapshot_cow() is not supported".to_string(),
        })
    }

    /// Restore this memory from a [`CowBacking`] captured by
    /// [`snapshot_cow`](LinearMemory::snapshot_cow).
    fn restore_cow(&mut self, _cow: &CowBacking) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedOperation {
            message: "restore_cow() is not supported".to_string(),
        })
    }

    /// Return a `VMMemoryDefinition` for exposing the memory to compiled wasm code.
    fn vmmemory(&self) -> NonNull<VMMemoryDefinition>;

    /// Attempts to clone this memory (if its cloneable)
    fn try_clone(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError>;

    #[doc(hidden)]
    /// # Safety
    /// This function is unsafe because WebAssembly specification requires that data is always set at initialization time.
    /// It should be the implementors responsibility to make sure this respects the spec
    unsafe fn initialize_with_data(&self, start: usize, data: &[u8]) -> Result<(), Trap> {
        unsafe {
            let memory = self.vmmemory().as_ref();

            initialize_memory_with_data(memory, start, data)
        }
    }

    /// Copies this memory to a new memory
    fn copy(&self) -> Result<Box<dyn LinearMemory + Send + Sync + 'static>, MemoryError>;

    /// Returns a concrete shared memory handle if this memory is shared.
    fn as_shared(&self) -> Result<VMSharedMemory, MemoryError> {
        Err(MemoryError::MemoryNotShared)
    }

    /// Add current thread to the waiter hash, and wait until notified or timeout.
    /// Return 0 if the waiter has been notified, 1 if there was a value mismatch,
    /// or 2 if the timeout occurred.
    ///
    /// # Safety
    /// the destination address must be a valid offset within this memory. It must also
    /// be properly aligned for the expected value type; either 4-byte aligned for
    /// `ExpectedValue::U32` or 8-byte aligned for `ExpectedValue::u64`.
    unsafe fn do_wait(
        &mut self,
        _dst: u32,
        _expected: ExpectedValue,
        _timeout: Option<Duration>,
    ) -> Result<u32, WaiterError> {
        Err(WaiterError::Unimplemented)
    }

    /// Notify waiters from the wait list. Return the number of waiters notified
    fn do_notify(&mut self, _dst: u32, _count: u32) -> u32 {
        0
    }

    /// Access the internal atomics handler.
    ///
    /// Will be [`None`] if the memory does not support atomics.
    fn thread_conditions(&self) -> Option<&ThreadConditions> {
        None
    }
}
