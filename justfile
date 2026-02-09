# Quick reference:
#   just              — type-check workspace (default)
#   just fmt          — format Rust code
#   just test         — run Rust tests
#   just py-test      — build extension + run Python tests
#   just ci           — full CI pipeline
#   just --list       — show all recipes

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

export UV_CACHE_DIR := justfile_directory() / ".uv-cache"

maturin := "uv run --with maturin maturin"

default: check

[group("rust")]
[doc("Format all Rust code")]
fmt:
    cargo fmt --all

[group("rust")]
[doc("Check Rust formatting without writing (CI-safe)")]
fmt-check:
    cargo fmt --all -- --check

[group("rust")]
[doc("Type-check the full workspace (tests, benches, examples)")]
check:
    cargo check --workspace --all-features --tests --benches --examples

[group("rust")]
[doc("Type-check workspace libraries only")]
check-lib:
    cargo check --workspace --all-features

[group("rust")]
[doc("Run clippy on all targets, deny warnings")]
clippy:
    cargo clippy --all --benches --tests --examples --all-features -- -D warnings

[group("rust")]
[doc("Run all Rust tests")]
test:
    cargo test --workspace --all-features

[group("rust")]
[doc("Build rustdoc for the workspace")]
doc:
    cargo doc --workspace --no-deps

[group("python")]
[doc("Sync Python dependencies")]
[working-directory: "bindings/python"]
py-sync:
    uv sync

[group("python")]
[doc("Build the Python extension in-place (dev mode)")]
[working-directory: "bindings/python"]
py-dev:
    {{ maturin }} develop

[group("python")]
[doc("Build a release wheel")]
[working-directory: "bindings/python"]
py-build:
    {{ maturin }} build --release

[group("python")]
[doc("Build dev extension then run Python tests")]
[working-directory: "bindings/python"]
py-test: py-dev
    uv run pytest

[group("python")]
[doc("Build dev extension then run documentation doctests")]
[working-directory: "bindings/python"]
docs-test: py-dev
    uv run pytest ../../scripts/test_docs_doctest.py -v

# ── Benchmarks ────────────────────────────────────────────────────────────

[group("bench")]
[doc("Run the default benchmark suite")]
bench-run:
    cargo run -p arco-bench -- run

[group("bench")]
[doc("Print a human-readable report from a results file")]
bench-report path:
    cargo run -p arco-bench -- report --input {{ path }}

[group("bench")]
[doc("Compare two benchmark result files")]
bench-compare baseline candidate:
    cargo run -p arco-bench -- compare \
        --baseline {{ baseline }} \
        --candidate {{ candidate }}

[group("bench")]
[doc("Gate CI on benchmark regressions (checks total + variables stages)")]
bench-gate baseline candidate duration_threshold="10" memory_threshold="10":
    #!/usr/bin/env bash
    set -euo pipefail
    for stage in total variables; do
        echo "Checking stage=${stage} duration<={{ duration_threshold }}% memory<={{ memory_threshold }}%"
        cargo run -p arco-bench -- compare \
            --baseline "{{ baseline }}" \
            --candidate "{{ candidate }}" \
            --stage "${stage}" \
            --duration-threshold-pct "{{ duration_threshold }}" \
            --memory-threshold-pct "{{ memory_threshold }}" \
            --format table
    done

[group("ci")]
[doc("Full CI pipeline: format → clippy → test → docs")]
ci: fmt-check clippy test docs-test
