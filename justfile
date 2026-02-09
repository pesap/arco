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

ci: fmt-check clippy test

# Python bindings

uv-sync:
	cd bindings && uv sync

python-dev:
	cd bindings/python && uv run maturin develop

python-build:
	cd bindings/python && uv run maturin build --release

python-test:
	cd bindings/python && maturin develop && uv run pytest
