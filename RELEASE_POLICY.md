# Release Policy

This repository publishes the Python package (`arco`) while keeping Rust crates
as internal implementation units. This document defines how versions and
release PRs are managed.

## Goals

- Keep user-facing releases predictable.
- Keep Python and Rust behavior aligned in every published version.
- Make it clear whether release changes came from Rust, Python, or both.

## Versioning Model

- We use one shared semantic version across:
  - `Cargo.toml` workspace version
  - `bindings/python/pyproject.toml` project version
  - `bindings/python/uv.lock` package entry for `arco`
- `release-please` manages version bumps and release PR creation.
- `linked-versions` keeps `arco-crates` and `arco-python` on the same version.

## Release PR Structure

- `release-please` opens separate PRs:
  - `arco-crates` PR (Rust workspace changes)
  - `arco-python` PR (Python binding/package changes)
- PRs stay separate for visibility, but versions stay synchronized.
- Notes are generated with the default changelog strategy so commit categories
  are preserved.

## Publishing Behavior

- `arco-python` is the published package (PyPI + GitHub release artifacts).
- `arco-crates` is internal (`skip-github-release: true`) and is used for
  version tracking and change visibility.

## How To Read Release PRs

- Rust-only change:
  - Expect both PRs at the same version due to linked versions.
  - `arco-crates` PR usually has most meaningful changelog entries.
- Python-only change:
  - Expect both PRs at the same version.
  - `arco-python` PR usually has most meaningful changelog entries.
- Mixed change:
  - Both PRs contain relevant entries.

## Commit Conventions

- Use Conventional Commits (`feat:`, `fix:`, `perf:`, `chore:`, etc.).
- Non-conventional commit messages may be skipped or poorly classified in
  release notes.

## Forcing A Release PR

If you need to force a release:

1. Create a conventional commit that touches `crates/**` or
   `bindings/python/**`.
2. Add a `Release-As` trailer to set the target version explicitly.

Example:

```text
chore(release): force 0.2.1

Release-As: 0.2.1
```

Because versions are linked, forcing one component release version will
synchronize the other component to the same version.
