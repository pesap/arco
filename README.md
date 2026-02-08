### arco

> A performant, memory efficient optimization library for power system problems
>
> [![Rust](https://img.shields.io/badge/rust-1.85-brightgreen)](https://www.rust-lang.org/)
> [![Docs](https://img.shields.io/badge/docs-di%C3%A1taxis-green)](docs/diataxis.md)
> [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
> [![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)

Arco is an experimental workspace for linear and mixed-integer optimization. The
primary user-facing API is the Python binding module `arco`, backed by Rust
crates for model construction, solver integration, and diagnostics.

> [!warning]
> arco is pretty new and it is not advised for production environments

## Python API quickstart

From the repository root:

```bash
cd bindings/python
uv sync --group dev
uv run maturin develop
uv run python examples/simple_lp.py
```

Minimal example using the current API:

```python
import arco

model = arco.Model()

x = model.add_variable(bounds=arco.NonNegativeFloat, name="x")
y = model.add_variable(bounds=arco.NonNegativeFloat, name="y")

model.add_constraint(x + y >= 5.0, name="demand")
model.minimize(3.0 * x + 2.0 * y, name="cost")

result = model.solve()
print(result.status)
print(result.objective_value)
print(result.get_value(x), result.get_value(y))
```

Two-block `BlockModel` example (each block builds one model):

```python
import arco


def make_model(lower: float, upper: float) -> arco.Model:
    model = arco.Model()
    x = model.add_variable(bounds=arco.Bounds(lower=lower, upper=upper), is_integer=False)
    model.minimize(expr=x)
    return model


def build_stage_one(ctx: arco.BlockContext) -> arco.Model:
    return make_model(lower=4.0, upper=10.0)


def extract_stage_one(solution: arco.Solution, _ctx: arco.BlockContext) -> dict[str, float]:
    return {"cap": solution.get_primal(index=0)}


def build_stage_two(ctx: arco.BlockContext) -> arco.Model:
    return make_model(lower=0.0, upper=ctx.inputs["cap"])


def extract_stage_two(solution: arco.Solution, _ctx: arco.BlockContext) -> dict[str, float]:
    return {"value": solution.get_primal(index=0)}


blocks = arco.BlockModel(name="two_stage")
stage_one = blocks.add_block(
    build_stage_one,
    name="stage_one",
    outputs={"cap": float},
    extract=extract_stage_one,
)
stage_two = blocks.add_block(
    build_stage_two,
    name="stage_two",
    inputs_schema={"cap": float},
    outputs={"value": float},
    extract=extract_stage_two,
)
blocks.link(source=stage_one.output("cap"), target=stage_two.input("cap"))

runs = blocks.solve()
for run in runs:
    print(run.name, run.solution.status, run.outputs)
```

## Python API overview

### Low-level API — single-model building and solving

Build one `Model`, add variables and constraints, set an objective, and solve.

| Area            | Surface                                                                                                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model lifecycle | `Model()` → add variables/constraints → `minimize()`/`maximize()` → `solve()`                                                                                            |
| Variables       | `add_variable(bounds, ...)` returns a `Variable`; `add_variables(index_sets, bounds, ...)` returns a `VariableArray`. Bound constants `NonNegativeFloat`, `Binary`, etc. |
| Expressions     | Operator overloading on `Variable` and `LinearExpr` (`3.0 * x + y`). Arithmetic on variables and scalars builds expressions directly                                     |
| Constraints     | `add_constraint(expr)` accepts comparisons (`x + y <= 10`); `add_constraints(expr)` for arrays. `add_eq()`, `add_le()`, `add_ge()` accept a `LinearExpr` and scalar rhs  |
| Objective       | `minimize(expr)` / `maximize(expr)`, or `set_objective(sense, terms)`                                                                                                    |
| Solve           | `solve()` returns `SolveResult` with `status`, `objective_value`, `get_value`, `get_dual`, `get_reduced_cost`, `get_slack`                                               |
| Solver backends | `HiGHS` (default) and `Xpress`. Pass `solver=arco.Xpress()` to `Model()` or `solve()`                                                                                    |
| Inspection      | `inspect()` returns a `ModelSnapshot`; `export_csc()` / `export_crs()` for sparse matrix export                                                                          |
| Slacks          | `add_slack()`, `add_slacks()`, `make_elastic()` for infeasibility diagnosis                                                                                              |

### High-level API — multi-model block composition

Compose multiple independent models into a DAG where outputs of one block feed
inputs of the next.

| Area       | Surface                                                                                                                                                                                             |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BlockModel | `BlockModel(name=...)` is the orchestrator                                                                                                                                                          |
| Blocks     | `add_block(build_fn, name=..., outputs=..., extract=...)` registers a callable that receives a `BlockContext` and returns a `Model`. The `extract` callable maps a `Solution` back to output values |
| Linking    | `link(source=handle.output("key"), target=handle.input("key"))` wires block outputs to downstream inputs                                                                                            |
| Solve      | `BlockModel.solve()` executes blocks in topological order and returns a list of `BlockRun` results                                                                                                  |
| Transforms | `Transform.scale()`, `Transform.offset()`, `Transform.shift()`, etc. can be piped (`\|`) and applied to arrays between blocks                                                                       |
| Context    | `BlockContext.inputs` provides upstream values; `BlockContext.attach(key, value)` stashes objects for extraction                                                                                    |

## Rust usage

The Rust workspace is organized around reusable crates:

- `arco-core` for model construction and lifecycle logic.
- `arco-highs` and `arco-xpress` for solver integrations.
- `arco-tools` for diagnostics and memory instrumentation.
- `arco-blocks` and `arco-blocks-core` for block-oriented composition.

See `docs/how-to/rust-quickstart.md` and `docs/explanation/02-architecture.md`
for Rust-first workflows.

## Performance notes

Performance and memory model details are documented in:

- `docs/explanation/09-performance-model.md`
- `scripts/model-bench`

Use `just model-bench` to run benchmark scenarios.

## Workspace layout

- `crates/` contains Rust workspace members.
- `bindings/python` contains the PyO3 extension and Python tests/examples.
- `docs/` follows a Diataxis documentation layout.
- `scripts/` contains benchmarking and tooling utilities.

## Development commands

- `just build` runs `cargo fmt`, `cargo clippy --all-targets --all-features`,
  and `cargo build`.
- `just test` runs Rust tests.
- `cd bindings/python && uv run pytest tests` runs Python binding tests.
- `cd bindings/python && uv run ruff check .` runs Python linting.
- `cd bindings/python && uv run ty check .` runs Python type checks.

## Documentation index

- `docs/diataxis.md` is the documentation entrypoint.
- `docs/rfd/README.md` contains architecture and design RFCs.
- `CONTRIBUTING.md` describes contribution and review expectations.
