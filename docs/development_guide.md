# Development Guide

## Development

- Format sources with `cargo fmt` before committing changes.
- Lint with `cargo clippy --all-targets --all-features` to catch obvious mistakes across sketches and orchestration layers.

## CI gates

Run these before pushing; they are exactly what `.github/workflows/ci.yml` and
`.github/workflows/docs.yml` enforce. The local commands in the section above
are looser than the gates — `cargo fmt` rewrites instead of failing, and clippy
without `--workspace --locked -- -D warnings` passes on code CI rejects.

Format:

```bash
cargo fmt --all -- --check
```

Clippy, once per feature set, warnings denied:

```bash
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo clippy --workspace --all-targets --features octo-runtime --locked -- -D warnings
cargo clippy --workspace --all-targets --features experimental --locked -- -D warnings
```

Tests, over the same four feature sets:

```bash
cargo test --all-features --locked
cargo test --locked
cargo test --features octo-runtime --locked
cargo test --features experimental --locked
```

Vendored proto code must have no drift — regenerate, then confirm the tree is
unchanged:

```bash
cargo run --manifest-path tools/gen-proto/Cargo.toml --locked
git diff --exit-code -- src/proto/generated
```

Rustdoc, warnings denied:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --locked
```

The Rust CI workflow skips `**/*.md` and `docs/**`; the docs workflow runs on
`docs/**`, `README.md`, `src/**`, `proto/**`, `tools/gen-proto/**` and the
manifests. A documentation-only change is therefore gated by rustdoc alone.
