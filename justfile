set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default: check

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

check:
	cargo check --workspace --all-features

clippy:
	cargo clippy --all --benches --tests --examples --all-features -- -D warnings

test:
	cargo test --workspace --all-features

ci: fmt-check clippy test
