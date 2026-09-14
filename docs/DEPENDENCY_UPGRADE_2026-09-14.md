# Dacely dependency refresh — 2026-09-14

The `feat/reset-all-globals-aarch64` branch merges upstream `main` at
`6a844cff9bd2eb391ab91b7e41fee94260f0ec46` (Wasmer 7.4.0). The existing
Dacely snapshot, bounded restore, pooled coroutine stack, typed async call,
non-exported global, and Singlepass feature changes remain in the branch.
The nested submodules match upstream's pins.

Registry dependencies were checked against crates.io's sparse index and
updated to their latest non-yanked stable releases. Rust is pinned to 1.98.0.
The workspace lockfile was refreshed. Two intentional compatibility cases:

- `bincode` stays at 2.0.1. The 3.0.0 crate's entire `src/lib.rs` is a
  `compile_error!`; it is not a usable implementation.
- `getrandom03` stays at 0.3.4. This wasm-only feature-unification dependency
  enables `wasm_js` for transitive consumers of getrandom 0.3. The primary
  getrandom dependency is 0.4.3; changing the alias to 0.4 would no longer
  configure the older transitive instance.

The custom typed async call uses upstream's `StoreContext::install`, whose
async branch preserves the installed write guard across suspension. A new
pooled typed-call test exercises yielding host imports. The subscription
transport implements the existing GraphQL connection interface for current
Tungstenite; a duplex-stream test checks text, ping/pong, and close frames.

Validation on aarch64 Linux with LLVM 22:

- `make lint` (YAML, C/C++, Rust formatting, workspace/CLI/fuzz Clippy, TOML).
- Wasmer CLI build with LLVM enabled.
- Wasmer API `current_store`, `instance_snapshot`, and `jspi_async`: 27 tests
  passed; one manual benchmark remains ignored.
- Backend integration: 825 default-feature and 719 no-default-feature tests;
  the million-request isolation soak also passed.

Tests and builds can emit upstream aarch64 inline-assembly warnings about
FFR clobbers, and a future-incompatibility notice for proc-macro-error2.
