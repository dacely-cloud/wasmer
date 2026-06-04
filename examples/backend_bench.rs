//! Backend benchmark: Singlepass vs Cranelift.
//!
//! Measures, for the same module, on each backend:
//!   * compile time            (wasm -> native code, `Module::new`)
//!   * serialized cache size   (`Module::serialize`)
//!   * cache-load time         (`Module::deserialize`, headless engine)
//!   * steady-state exec time  (typed call of a compute+memory+loop kernel)
//!
//! The kernel is a data-dependent LCG chain that, every iteration, also does an
//! 8-byte load + store at a rotating offset over a 1 MiB linear memory. The
//! dependent chain limits instruction-level parallelism, so codegen quality
//! (register allocation, instruction selection) shows up clearly.
//!
//! Run:
//!   cargo run --release --example backend-bench --features "singlepass cranelift" -- [N] [reps]

use std::time::{Duration, Instant};

use wasmer::{Instance, Module, Store, TypedFunction, imports, sys::EngineBuilder, wat2wasm};
use wasmer_compiler_cranelift::Cranelift;
#[cfg(feature = "llvm")]
use wasmer_compiler_llvm::LLVM;
use wasmer_compiler_singlepass::Singlepass;

const WAT: &str = r#"
(module
  (memory (export "mem") 16)            ;; 16 pages = 1 MiB
  (func (export "run") (param $n i64) (result i64)
    (local $i i64) (local $acc i64) (local $addr i32)
    (local.set $acc (i64.const 1))
    (block $done
      (loop $loop
        (br_if $done (i64.ge_u (local.get $i) (local.get $n)))
        ;; addr = (i << 3) & 0xFFFF8  -> 8-byte aligned, within 1 MiB
        (local.set $addr
          (i32.and
            (i32.wrap_i64 (i64.shl (local.get $i) (i64.const 3)))
            (i32.const 0xFFFF8)))
        ;; acc = rotl(acc ^ mem[addr], 23) * LCG_MUL + i   (non-degenerate hash)
        (local.set $acc
          (i64.add
            (i64.mul
              (i64.rotl
                (i64.xor (local.get $acc) (i64.load (local.get $addr)))
                (i64.const 23))
              (i64.const 6364136223846793005))
            (local.get $i)))
        ;; mem[addr] = acc   (load + store every iteration)
        (i64.store (local.get $addr) (local.get $acc))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $acc)))
"#;

/// Vectorizable kernel: a pure integer-polynomial reduction. Each iteration's
/// term depends ONLY on `i` (independent across iterations), and the running
/// total is an integer-add reduction (associative -> can use partial-sum
/// vectors). This is the textbook auto-vectorizable shape: an optimizer with a
/// loop vectorizer can compute 4-8 lanes per step (AVX2/AVX-512) and reduce at
/// the end. No memory traffic, so it is compute-bound.
const WAT_VEC: &str = r#"
(module
  (func (export "run") (param $n i64) (result i64)
    (local $i i64) (local $sum i64)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_u (local.get $i) (local.get $n)))
        ;; sum += i*i*C1 + i*C2 + C3   (independent per-i term)
        (local.set $sum
          (i64.add (local.get $sum)
            (i64.add
              (i64.add
                (i64.mul (i64.mul (local.get $i) (local.get $i)) (i64.const 2654435761))
                (i64.mul (local.get $i) (i64.const 40503)))
              (i64.const 12345))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $sum)))
"#;

/// Data-dependent vectorizable kernel: a reduction that loads from memory each
/// iteration. The load defeats closed-form/scalar-evolution loop elimination
/// (the compiler can't know memory contents), so the loop MUST actually run all
/// N iterations. But iterations are still independent and it's an integer-add
/// reduction, so it IS auto-vectorizable. This isolates "does the backend turn
/// a scalar loop into SIMD?" from "can it delete the loop entirely?".
const WAT_VEC_MEM: &str = r#"
(module
  (memory (export "mem") 16)
  (func (export "run") (param $n i64) (result i64)
    (local $i i64) (local $sum i64) (local $addr i32)
    (block $done
      (loop $loop
        (br_if $done (i64.ge_u (local.get $i) (local.get $n)))
        (local.set $addr
          (i32.and
            (i32.wrap_i64 (i64.shl (local.get $i) (i64.const 3)))
            (i32.const 0xFFFF8)))
        ;; sum += mem[addr] + i   (load -> not closed-form; independent -> SIMD-able)
        (local.set $sum
          (i64.add (local.get $sum)
            (i64.add (i64.load (local.get $addr)) (local.get $i))))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $loop)))
    (local.get $sum)))
"#;

/// Run `f` `reps` times, return the *minimum* elapsed time and one sample result.
/// Minimum is the right statistic for compute microbenchmarks: it filters
/// scheduler/cache noise and approximates the no-interference cost.
fn time_min<T>(reps: usize, mut f: impl FnMut() -> T) -> (Duration, T) {
    let mut best = Duration::MAX;
    let mut sample = f(); // warm up; result overwritten below
    for _ in 0..reps {
        let start = Instant::now();
        let r = f();
        let dt = start.elapsed();
        if dt < best {
            best = dt;
        }
        sample = r;
    }
    (best, sample)
}

struct Report {
    name: &'static str,
    compile: Duration,
    cache_bytes: usize,
    deserialize: Duration,
    exec: Duration,
    result: i64,
}

fn run_backend(
    name: &'static str,
    wasm: &[u8],
    n: i64,
    exec_reps: usize,
    make_store: impl Fn() -> Store,
) -> Report {
    // --- compile time: fresh store + engine each rep, throwaway modules ---
    let (compile, _) = time_min(5, || {
        let store = make_store();
        Module::new(&store, wasm).expect("compile failed")
    });

    // Persistent store + module used for serialization and execution.
    let mut store = make_store();
    let module = Module::new(&store, wasm).expect("compile failed");

    // --- serialize (build the on-disk/in-memory cache) ---
    let cache = module.serialize().expect("serialize failed");
    let cache_bytes = cache.len();

    // --- deserialize (load cache) into a headless engine (no compiler) ---
    let (deserialize, _) = time_min(5, || {
        let hstore = Store::new(EngineBuilder::headless());
        // Safety: bytes were just produced by serialize() on this host.
        unsafe { Module::deserialize(&hstore, cache.clone()).expect("deserialize failed") }
    });

    // --- execution: instantiate once, warm up, time the typed call ---
    let instance = Instance::new(&mut store, &module, &imports! {}).expect("instantiate failed");
    let run: TypedFunction<i64, i64> = instance
        .exports
        .get_function("run")
        .unwrap()
        .typed(&store)
        .unwrap();

    let _ = run.call(&mut store, n).unwrap(); // warm caches/JIT pages
    let (exec, result) = time_min(exec_reps, || run.call(&mut store, n).unwrap());

    Report {
        name,
        compile,
        cache_bytes,
        deserialize,
        exec,
        result,
    }
}

/// Generate a module with `num_funcs` independent compute-loop functions, to
/// show how compile time scales with module size (a one-function module hides
/// the gap because per-function optimization cost dominates only in bulk).
fn gen_module(num_funcs: usize) -> Vec<u8> {
    let mut s = String::from("(module\n  (memory 1)\n");
    for k in 0..num_funcs {
        s.push_str(&format!(
            "  (func (export \"f{k}\") (param $n i64) (result i64)\n\
             \x20   (local $i i64) (local $a i64)\n\
             \x20   (local.set $a (i64.const {seed}))\n\
             \x20   (block $d (loop $l\n\
             \x20     (br_if $d (i64.ge_u (local.get $i) (local.get $n)))\n\
             \x20     (local.set $a (i64.xor\n\
             \x20       (i64.add (i64.mul (local.get $a) (i64.const 6364136223846793005))\n\
             \x20                (i64.const 1442695040888963407))\n\
             \x20       (local.get $i)))\n\
             \x20     (local.set $i (i64.add (local.get $i) (i64.const 1))) (br $l)))\n\
             \x20   (local.get $a))\n",
            k = k,
            seed = k as u64 + 1,
        ));
    }
    s.push_str(")\n");
    wat2wasm(s.as_bytes()).expect("wat2wasm").to_vec()
}

fn compile_min(wasm: &[u8], make_store: impl Fn() -> Store) -> Duration {
    time_min(5, || {
        let store = make_store();
        Module::new(&store, wasm).expect("compile failed")
    })
    .0
}

/// Run every available backend on one kernel and print a table + summary.
fn compare(title: &str, wasm: &[u8], n: i64, reps: usize) {
    println!("========== {title} ==========");

    // LLVM is included only when compiled with `--features llvm`.
    let mut reports = vec![
        run_backend("Singlepass", wasm, n, reps, || Store::new(Singlepass::default())),
        run_backend("Cranelift", wasm, n, reps, || Store::new(Cranelift::default())),
    ];
    #[cfg(feature = "llvm")]
    reports.push(run_backend("LLVM", wasm, n, reps, || Store::new(LLVM::default())));

    // Correctness: every backend must agree on the result.
    let baseline_result = reports[0].result;
    for r in &reports {
        assert_eq!(r.result, baseline_result, "{} disagrees on result!", r.name);
    }
    println!(
        "result check: all {} backends returned {:#018x}\n",
        reports.len(),
        baseline_result
    );

    let ns_per_iter = |d: Duration| d.as_secs_f64() * 1e9 / n as f64;

    println!(
        "{:<12} {:>16} {:>16} {:>12}",
        "backend", "exec (min)", "ns/iter", "vs Singlepass"
    );
    println!("{}", "-".repeat(60));
    let base_exec = reports[0].exec.as_secs_f64();
    for r in &reports {
        println!(
            "{:<12} {:>13.3} ms {:>12.3} {:>11.2}x",
            r.name,
            r.exec.as_secs_f64() * 1e3,
            ns_per_iter(r.exec),
            base_exec / r.exec.as_secs_f64(),
        );
    }
    // Direct optimizer-vs-optimizer comparison (the interesting one here).
    #[cfg(feature = "llvm")]
    if reports.len() == 3 {
        println!(
            "\n  LLVM vs Cranelift: {:.2}x   (>1 means LLVM faster)",
            reports[1].exec.as_secs_f64() / reports[2].exec.as_secs_f64()
        );
    }
    println!();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: i64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60_000_001);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(9);

    println!("n = {n} iterations, exec reps = {reps} (min reported)\n");

    let serial = wat2wasm(WAT.as_bytes()).expect("wat2wasm");
    let vec_k = wat2wasm(WAT_VEC.as_bytes()).expect("wat2wasm");

    compare(
        "Kernel A: serial dependent chain (NOT vectorizable)",
        &serial,
        n,
        reps,
    );
    compare(
        "Kernel B: independent polynomial reduction (closed-form-able)",
        &vec_k,
        n,
        reps,
    );
    let vec_mem = wat2wasm(WAT_VEC_MEM.as_bytes()).expect("wat2wasm");
    compare(
        "Kernel C: data-dependent reduction (vectorizable, NOT closed-form)",
        &vec_mem,
        n,
        reps,
    );

    // --- compile-time scaling vs module size ---
    println!("\ncompile time vs module size (functions x compute loop):");
    #[cfg(feature = "llvm")]
    println!(
        "{:<12} {:>13} {:>13} {:>13} {:>9} {:>9}",
        "functions", "Singlepass", "Cranelift", "LLVM", "CL/SP", "LLVM/SP"
    );
    #[cfg(not(feature = "llvm"))]
    println!(
        "{:<12} {:>13} {:>13} {:>9}",
        "functions", "Singlepass", "Cranelift", "CL/SP"
    );
    println!("{}", "-".repeat(72));
    for &k in &[1usize, 64, 256, 1024] {
        let m = gen_module(k);
        let sp_c = compile_min(&m, || Store::new(Singlepass::default()));
        let cl_c = compile_min(&m, || Store::new(Cranelift::default()));
        #[cfg(feature = "llvm")]
        {
            let llvm_c = compile_min(&m, || Store::new(LLVM::default()));
            println!(
                "{:<12} {:>10.3} ms {:>10.3} ms {:>10.3} ms {:>8.1}x {:>8.1}x",
                k,
                sp_c.as_secs_f64() * 1e3,
                cl_c.as_secs_f64() * 1e3,
                llvm_c.as_secs_f64() * 1e3,
                cl_c.as_secs_f64() / sp_c.as_secs_f64(),
                llvm_c.as_secs_f64() / sp_c.as_secs_f64(),
            );
        }
        #[cfg(not(feature = "llvm"))]
        println!(
            "{:<12} {:>10.3} ms {:>10.3} ms {:>8.1}x",
            k,
            sp_c.as_secs_f64() * 1e3,
            cl_c.as_secs_f64() * 1e3,
            cl_c.as_secs_f64() / sp_c.as_secs_f64(),
        );
    }
}
