set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default: check

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

check:
	cargo check --workspace --all-features --tests --benches --examples

check-lib:
	cargo check --workspace --all-features

clippy:
	cargo clippy --all --benches --tests --examples --all-features -- -D warnings

test:
	cargo test --workspace --all-features

doc:
	cargo doc --workspace --no-deps

ci: fmt-check clippy test docs-test

# Benchmarks

bench-run:
	cargo run -p arco-bench -- run

bench-report path:
	cargo run -p arco-bench -- report --input {{path}}

bench-compare baseline candidate:
	cargo run -p arco-bench -- compare --baseline {{baseline}} --candidate {{candidate}}

bench-gate baseline candidate duration_threshold="10" memory_threshold="10":
	./scripts/bench_gate.sh {{baseline}} {{candidate}} {{duration_threshold}} {{memory_threshold}}

# Python bindings

uv-sync:
	cd bindings && UV_CACHE_DIR=../.uv-cache uv sync

python-dev:
	cd bindings/python && UV_CACHE_DIR=../../.uv-cache uv run --with maturin maturin develop

python-build:
	cd bindings/python && UV_CACHE_DIR=../../.uv-cache uv run --with maturin maturin build --release

python-test:
	cd bindings/python && UV_CACHE_DIR=../../.uv-cache uv run --with maturin maturin develop && UV_CACHE_DIR=../../.uv-cache uv run pytest

docs-test:
	cd bindings/python && UV_CACHE_DIR=../../.uv-cache uv run --with maturin maturin develop && UV_CACHE_DIR=../../.uv-cache uv run pytest ../../scripts/test_docs_doctest.py -v
