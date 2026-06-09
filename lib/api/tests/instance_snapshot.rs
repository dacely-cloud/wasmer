//! Tests for `Instance::snapshot` / `Instance::reset_to_snapshot` (sys backend).
//!
//! These exercise the warm-instance reuse pattern: capture a post-instantiation
//! baseline, mutate memory / globals / tables / passive segments, then restore
//! and assert the instance is byte-for-byte back to baseline with no state
//! leaking across reuses.
#![cfg(feature = "sys")]

use wasmer::{Instance, Memory, Module, Store, imports};

/// A module that exposes a mutable global, a linear memory, and a funcref table,
/// plus functions to mutate and observe each.
const WAT: &str = r#"
(module
  (type $sig (func (result i32)))
  (memory (export "mem") 1 10)
  (global $g (export "g") (mut i32) (i32.const 7))
  (table $t (export "t") 3 funcref)
  (elem (i32.const 0) $f0)
  (elem declare func $f1)
  (func $f0 (result i32) (i32.const 100))
  (func $f1 (result i32) (i32.const 200))

  (func (export "write_byte") (param $addr i32) (param $val i32)
    (i32.store8 (local.get $addr) (local.get $val)))
  (func (export "read_byte") (param $addr i32) (result i32)
    (i32.load8_u (local.get $addr)))
  (func (export "set_g") (param $v i32)
    (global.set $g (local.get $v)))
  (func (export "get_g") (result i32)
    (global.get $g))
  (func (export "grow") (param $p i32) (result i32)
    (memory.grow (local.get $p)))
  (func (export "size") (result i32)
    (memory.size))
  (func (export "set_slot1")
    (table.set $t (i32.const 1) (ref.func $f1)))
  (func (export "slot_is_null") (param $i i32) (result i32)
    (ref.is_null (table.get $t (local.get $i))))
  (func (export "set_slot0_f1")
    (table.set $t (i32.const 0) (ref.func $f1)))
  (func (export "call0") (result i32)
    (call_indirect $t (type $sig) (i32.const 0)))
)
"#;

#[test]
fn table_reset_visible_to_call_indirect() {
    // Regression: `call_indirect` dispatches through the vmctx inline
    // fixed-funcref array, NOT the `VMTable` backing vec. Restoring only the
    // backing (or a `Value` round-trip) leaves a `table.set` visible to indirect
    // calls; reset must re-sync the inline array.
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let call0 = exports.get_typed_function::<(), i32>(&store, "call0").unwrap();
    let set_slot0 = exports
        .get_typed_function::<(), ()>(&store, "set_slot0_f1")
        .unwrap();

    let snap = instance.snapshot(&store).unwrap();
    assert_eq!(call0.call(&mut store).unwrap(), 100, "baseline: slot 0 = $f0");

    set_slot0.call(&mut store).unwrap();
    assert_eq!(call0.call(&mut store).unwrap(), 200, "mutated: slot 0 = $f1");

    instance.reset_to_snapshot(&mut store, &snap).unwrap();
    assert_eq!(
        call0.call(&mut store).unwrap(),
        100,
        "reset must restore slot 0 as seen by call_indirect"
    );
}

#[test]
fn snapshot_round_trip_memory_global_table() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();

    let exports = &instance.exports;
    let write = exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let read = exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();
    let set_g = exports.get_typed_function::<i32, ()>(&store, "set_g").unwrap();
    let get_g = exports.get_typed_function::<(), i32>(&store, "get_g").unwrap();
    let grow = exports.get_typed_function::<i32, i32>(&store, "grow").unwrap();
    let size = exports.get_typed_function::<(), i32>(&store, "size").unwrap();
    let set_slot1 = exports
        .get_typed_function::<(), ()>(&store, "set_slot1")
        .unwrap();
    let slot_is_null = exports
        .get_typed_function::<i32, i32>(&store, "slot_is_null")
        .unwrap();

    // Capture the pristine baseline.
    let snap = instance.snapshot(&store).unwrap();

    // Mutate every kind of state.
    set_g.call(&mut store, 42).unwrap();
    write.call(&mut store, 16, 0xAB).unwrap();
    let _ = grow.call(&mut store, 2).unwrap(); // 1 -> 3 pages
    set_slot1.call(&mut store).unwrap();

    assert_eq!(get_g.call(&mut store).unwrap(), 42);
    assert_eq!(read.call(&mut store, 16).unwrap(), 0xAB);
    assert_eq!(size.call(&mut store).unwrap(), 3);
    assert_eq!(slot_is_null.call(&mut store, 1).unwrap(), 0); // set to $f1

    // Restore to baseline.
    instance.reset_to_snapshot(&mut store, &snap).unwrap();

    assert_eq!(get_g.call(&mut store).unwrap(), 7, "global restored");
    assert_eq!(read.call(&mut store, 16).unwrap(), 0, "memory restored");
    assert_eq!(size.call(&mut store).unwrap(), 1, "memory size restored");
    assert_eq!(
        slot_is_null.call(&mut store, 1).unwrap(),
        1,
        "table slot 1 cleared back to null"
    );
    assert_eq!(
        slot_is_null.call(&mut store, 0).unwrap(),
        0,
        "table slot 0 still holds the active elem funcref"
    );
}

#[test]
fn grown_pages_are_zeroed_after_reset() {
    // Regression guard: after a reset that shrinks the memory, a later grow back
    // into the previously-used region must observe zero-initialized memory (Wasm
    // spec), not stale bytes from the prior run.
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let write = exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let read = exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();
    let grow = exports.get_typed_function::<i32, i32>(&store, "grow").unwrap();

    let snap = instance.snapshot(&store).unwrap();

    // Grow, dirty a byte high in the new page, then reset.
    let _ = grow.call(&mut store, 1).unwrap(); // 1 -> 2 pages
    let high_addr = 65536 + 100; // inside page 2
    write.call(&mut store, high_addr, 0xFF).unwrap();
    instance.reset_to_snapshot(&mut store, &snap).unwrap();

    // Grow again and read the same address: must be zero.
    let _ = grow.call(&mut store, 1).unwrap();
    assert_eq!(
        read.call(&mut store, high_addr).unwrap(),
        0,
        "regrown page must be zero, not stale prior-run data"
    );
}

#[test]
fn snapshot_isolation_across_reuses() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let write = exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let read = exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();
    let set_g = exports.get_typed_function::<i32, ()>(&store, "set_g").unwrap();
    let get_g = exports.get_typed_function::<(), i32>(&store, "get_g").unwrap();

    let snap = instance.snapshot(&store).unwrap();

    // Tenant A runs and leaves state behind.
    set_g.call(&mut store, 111).unwrap();
    write.call(&mut store, 0, 0x11).unwrap();
    instance.reset_to_snapshot(&mut store, &snap).unwrap();

    // Tenant B must see none of tenant A's state.
    assert_eq!(get_g.call(&mut store).unwrap(), 7, "no global bleed");
    assert_eq!(read.call(&mut store, 0).unwrap(), 0, "no memory bleed");
    set_g.call(&mut store, 222).unwrap();
    write.call(&mut store, 0, 0x22).unwrap();
    instance.reset_to_snapshot(&mut store, &snap).unwrap();

    // Clean again for tenant C.
    assert_eq!(get_g.call(&mut store).unwrap(), 7);
    assert_eq!(read.call(&mut store, 0).unwrap(), 0);
}

/// A module using a passive data segment + `memory.init` / `data.drop`, to
/// verify reset re-populates passive segments (undoing `data.drop`).
const WAT_BULK: &str = r#"
(module
  (memory (export "mem") 1)
  (data $d "hello")
  (func (export "init_at") (param $dst i32)
    (memory.init $d (local.get $dst) (i32.const 0) (i32.const 5)))
  (func (export "drop_d")
    (data.drop $d))
  (func (export "read_byte") (param $a i32) (result i32)
    (i32.load8_u (local.get $a)))
)
"#;

#[test]
fn reset_restores_dropped_passive_data() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT_BULK).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let init_at = exports
        .get_typed_function::<i32, ()>(&store, "init_at")
        .unwrap();
    let drop_d = exports.get_typed_function::<(), ()>(&store, "drop_d").unwrap();
    let read = exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();

    let snap = instance.snapshot(&store).unwrap();

    // Use then drop the passive segment.
    init_at.call(&mut store, 0).unwrap();
    assert_eq!(read.call(&mut store, 0).unwrap(), b'h' as i32);
    drop_d.call(&mut store).unwrap();

    // Reset: memory cleared AND the dropped segment re-populated.
    instance.reset_to_snapshot(&mut store, &snap).unwrap();
    assert_eq!(read.call(&mut store, 0).unwrap(), 0, "memory cleared");

    // The segment is usable again after reset.
    init_at.call(&mut store, 0).unwrap();
    assert_eq!(
        read.call(&mut store, 0).unwrap(),
        b'h' as i32,
        "passive data segment restored after data.drop + reset"
    );
}

#[test]
fn reset_without_mutation_is_a_noop() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let get_g = instance
        .exports
        .get_typed_function::<(), i32>(&store, "get_g")
        .unwrap();
    let read = instance
        .exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();

    let snap = instance.snapshot(&store).unwrap();
    // Reset twice with no intervening mutation; state must stay at baseline.
    for _ in 0..2 {
        instance.reset_to_snapshot(&mut store, &snap).unwrap();
        assert_eq!(get_g.call(&mut store).unwrap(), 7);
        assert_eq!(read.call(&mut store, 16).unwrap(), 0);
    }
}

#[test]
fn snapshot_captures_a_non_pristine_baseline() {
    // The baseline is wherever you snapshot — not module-init. Mutate to a
    // "configured" state, snapshot, mutate further, then reset: state returns to
    // the snapshot point, not to the original instantiation.
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let set_g = instance
        .exports
        .get_typed_function::<i32, ()>(&store, "set_g")
        .unwrap();
    let get_g = instance
        .exports
        .get_typed_function::<(), i32>(&store, "get_g")
        .unwrap();
    let write = instance
        .exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let read = instance
        .exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();

    set_g.call(&mut store, 50).unwrap();
    write.call(&mut store, 8, 0xCD).unwrap();
    let snap = instance.snapshot(&store).unwrap(); // baseline := configured state

    set_g.call(&mut store, 99).unwrap();
    write.call(&mut store, 8, 0xEE).unwrap();

    instance.reset_to_snapshot(&mut store, &snap).unwrap();
    assert_eq!(get_g.call(&mut store).unwrap(), 50, "restored to snapshot, not init");
    assert_eq!(read.call(&mut store, 8).unwrap(), 0xCD);
}

#[test]
fn repeated_reset_loop_is_stable() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let set_g = exports.get_typed_function::<i32, ()>(&store, "set_g").unwrap();
    let get_g = exports.get_typed_function::<(), i32>(&store, "get_g").unwrap();
    let write = exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let read = exports
        .get_typed_function::<i32, i32>(&store, "read_byte")
        .unwrap();
    let grow = exports.get_typed_function::<i32, i32>(&store, "grow").unwrap();
    let size = exports.get_typed_function::<(), i32>(&store, "size").unwrap();
    let set_slot1 = exports
        .get_typed_function::<(), ()>(&store, "set_slot1")
        .unwrap();
    let slot_is_null = exports
        .get_typed_function::<i32, i32>(&store, "slot_is_null")
        .unwrap();

    let snap = instance.snapshot(&store).unwrap();

    for i in 0..64i32 {
        // Mutate every kind of state with iteration-dependent values.
        set_g.call(&mut store, i.wrapping_mul(7) + 1).unwrap();
        write.call(&mut store, (i % 100) * 4, (i & 0xff) as i32).unwrap();
        let _ = grow.call(&mut store, (i % 3) + 1).unwrap();
        set_slot1.call(&mut store).unwrap();

        instance.reset_to_snapshot(&mut store, &snap).unwrap();

        assert_eq!(get_g.call(&mut store).unwrap(), 7, "iter {i}: global");
        assert_eq!(read.call(&mut store, (i % 100) * 4).unwrap(), 0, "iter {i}: mem");
        assert_eq!(size.call(&mut store).unwrap(), 1, "iter {i}: size");
        assert_eq!(slot_is_null.call(&mut store, 1).unwrap(), 1, "iter {i}: table");
    }
}

/// Module with non-i32 globals, to confirm the raw `u128` capture round-trips
/// i64 / f64 globals correctly.
const WAT_TYPES: &str = r#"
(module
  (global $gi (export "gi") (mut i64) (i64.const 0))
  (global $gf (export "gf") (mut f64) (f64.const 0))
  (func (export "set_gi") (param i64) (global.set $gi (local.get 0)))
  (func (export "get_gi") (result i64) (global.get $gi))
  (func (export "set_gf") (param f64) (global.set $gf (local.get 0)))
  (func (export "get_gf") (result f64) (global.get $gf))
)
"#;

#[test]
fn restores_i64_and_f64_globals() {
    let mut store = Store::default();
    let module = Module::new(&store, WAT_TYPES).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let exports = &instance.exports;
    let set_gi = exports.get_typed_function::<i64, ()>(&store, "set_gi").unwrap();
    let get_gi = exports.get_typed_function::<(), i64>(&store, "get_gi").unwrap();
    let set_gf = exports.get_typed_function::<f64, ()>(&store, "set_gf").unwrap();
    let get_gf = exports.get_typed_function::<(), f64>(&store, "get_gf").unwrap();

    let snap = instance.snapshot(&store).unwrap();

    set_gi.call(&mut store, -0x0123_4567_89ab_cdef).unwrap();
    set_gf.call(&mut store, 3.141592653589793).unwrap();
    assert_eq!(get_gi.call(&mut store).unwrap(), -0x0123_4567_89ab_cdef);

    instance.reset_to_snapshot(&mut store, &snap).unwrap();
    assert_eq!(get_gi.call(&mut store).unwrap(), 0, "i64 global restored");
    assert_eq!(get_gf.call(&mut store).unwrap(), 0.0, "f64 global restored");
}

/// Grow `instance`'s memory to `pages` and fill it with a pattern — i.e. rebuild
/// a warmed baseline state. Used by the benchmark to make both the "reset" and
/// "re-instantiate" paths end at the *same* heap state (a fair comparison).
fn warm_to_baseline(store: &mut Store, instance: &Instance, pages: u32) {
    let grow = instance
        .exports
        .get_typed_function::<i32, i32>(store, "grow")
        .unwrap();
    let write = instance
        .exports
        .get_typed_function::<(i32, i32), ()>(store, "write_byte")
        .unwrap();
    if pages > 1 {
        let _ = grow.call(store, (pages - 1) as i32).unwrap();
    }
    let total = (pages * 65536) as i32;
    let mut a = 0;
    while a < total {
        write.call(store, a, a & 0xff).unwrap();
        a += 64;
    }
}

fn bench_one(label: &str, pages: u32) {
    use std::time::Instant;

    let mut store = Store::default();
    let module = Module::new(&store, WAT).unwrap();

    // Warmed baseline instance + snapshot.
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    warm_to_baseline(&mut store, &instance, pages);
    let snap = instance.snapshot(&store).unwrap();
    let set_g = instance
        .exports
        .get_typed_function::<i32, ()>(&store, "set_g")
        .unwrap();

    // Reuse path: dirty a global + reset the warmed instance.
    const N_RESET: usize = 2000;
    let t0 = Instant::now();
    for i in 0..N_RESET {
        set_g.call(&mut store, i as i32).unwrap();
        instance.reset_to_snapshot(&mut store, &snap).unwrap();
    }
    let reset_ns = t0.elapsed().as_nanos() / N_RESET as u128;

    // No-reuse path: re-instantiate AND rebuild the same baseline state — the
    // true alternative to resetting a warm instance.
    const N_REINIT: usize = 100;
    let t1 = Instant::now();
    for _ in 0..N_REINIT {
        let inst = Instance::new(&mut store, &module, &imports! {}).unwrap();
        warm_to_baseline(&mut store, &inst, pages);
        std::hint::black_box(&inst);
    }
    let reinit_ns = t1.elapsed().as_nanos() / N_REINIT as u128;

    println!(
        "[{label}]  reset = {reset_ns:>7} ns/op   reinit+rebuild = {reinit_ns:>8} ns/op   speedup = {:.2}x",
        reinit_ns as f64 / reset_ns as f64
    );
}

/// Rough latency comparison: resetting a warm instance vs re-instantiating and
/// rebuilding the same baseline. Both paths end at identical state.
///
/// Not a statistical benchmark — run with:
///   cargo test --release -p wasmer --features sys,cranelift,wat \
///     --test instance_snapshot -- --ignored --nocapture bench_reset_vs_reinstantiate
#[test]
#[ignore = "benchmark; run with --ignored --nocapture"]
fn bench_reset_vs_reinstantiate() {
    println!("\n--- warm-instance reset vs re-instantiate+rebuild (same end state) ---");
    bench_one("1 page  (64 KiB) ", 1);
    bench_one("9 pages (576 KiB)", 9);
    println!(
        "Eager reset is O(heap); the memfd CoW fast path (Linux, static memory) is O(dirty pages).\n"
    );
}

/// Read a memory's full accessible contents.
fn read_all(store: &Store, mem: &Memory) -> Vec<u8> {
    let view = mem.view(store);
    let len = view.data_size() as usize;
    let mut buf = vec![0u8; len];
    view.read(0, &mut buf).unwrap();
    buf
}

/// A minimal module with a *dynamic* (no-maximum) memory, which forces the eager
/// restore path (CoW is only used for static-style memories).
const WAT_DYN: &str = r#"
(module
  (memory (export "mem") 1)
  (func (export "write_byte") (param $a i32) (param $v i32)
    (i32.store8 (local.get $a) (local.get $v)))
  (func (export "read_byte") (param $a i32) (result i32)
    (i32.load8_u (local.get $a)))
  (func (export "grow") (param $p i32) (result i32) (memory.grow (local.get $p)))
  (func (export "size") (result i32) (memory.size))
)
"#;

/// Build a 3-page patterned baseline, snapshot, scatter-mutate + grow, reset,
/// then assert the full memory image is byte-for-byte back to the baseline.
fn assert_byte_exact_restore(wat: &str) {
    let mut store = Store::default();
    let module = Module::new(&store, wat).unwrap();
    let instance = Instance::new(&mut store, &module, &imports! {}).unwrap();
    let mem = instance.exports.get_memory("mem").unwrap().clone();
    let write = instance
        .exports
        .get_typed_function::<(i32, i32), ()>(&store, "write_byte")
        .unwrap();
    let grow = instance.exports.get_typed_function::<i32, i32>(&store, "grow").unwrap();
    let size = instance.exports.get_typed_function::<(), i32>(&store, "size").unwrap();

    // Patterned 3-page baseline.
    let _ = grow.call(&mut store, 2).unwrap();
    let mut a = 0;
    while a < 3 * 65536 {
        write.call(&mut store, a, (a >> 6) & 0xff).unwrap();
        a += 64;
    }
    let baseline = read_all(&store, &mem);
    assert_eq!(baseline.len(), 3 * 65536);

    let snap = instance.snapshot(&store).unwrap();

    // Scatter-overwrite the baseline pages, then grow and dirty new pages.
    let mut a = 0;
    while a < 3 * 65536 {
        write.call(&mut store, a, 0xEE).unwrap();
        a += 64;
    }
    let _ = grow.call(&mut store, 2).unwrap();
    let mut a = 3 * 65536;
    while a < 5 * 65536 {
        write.call(&mut store, a, 0xCC).unwrap();
        a += 64;
    }

    instance.reset_to_snapshot(&mut store, &snap).unwrap();

    assert_eq!(size.call(&mut store).unwrap(), 3, "size restored to baseline");
    assert_eq!(read_all(&store, &mem), baseline, "memory restored byte-for-byte");
}

#[test]
fn cow_path_restore_is_byte_exact() {
    // Static memory (max set) -> memfd CoW path on Linux.
    assert_byte_exact_restore(WAT);
}

#[test]
fn eager_path_restore_is_byte_exact() {
    // Dynamic memory (no max) -> eager memcpy path everywhere.
    assert_byte_exact_restore(WAT_DYN);
}

#[test]
fn cow_disabled_env_forces_eager_path() {
    // The escape hatch must still restore correctly (both paths are equivalent).
    // SAFETY: single-threaded within this test; other tests are correct under
    // either path, so transient visibility of the var does not affect them.
    unsafe { std::env::set_var("WASMER_DISABLE_INSTANCE_COW", "1") };
    assert_byte_exact_restore(WAT);
    unsafe { std::env::remove_var("WASMER_DISABLE_INSTANCE_COW") };
}
