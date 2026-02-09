# Block Composition

This tutorial shows you how to decompose a large optimization problem into
independent blocks that arco solves in sequence, passing results from one block
to the next.

## Why blocks?

Large optimization models tend to grow into monoliths. A single function
creates hundreds of variables, adds dozens of constraint families, and wires
everything to one objective. When something breaks, it is hard to tell which
piece went wrong. When only part of the model needs to change, you still have
to understand the whole thing.

Blocks let you split a model into self-contained pieces. Each block has its own
build function, its own variables and constraints, and its own solve call. Arco
manages the execution order, passes outputs from upstream blocks to downstream
blocks through typed ports, and collects per-block results so you can inspect
each piece independently.

## Defining a block

A block is a callable that receives a `BlockContext` and returns an
`arco.Model`. The context carries input data through `ctx.inputs`, a dictionary
whose keys are the input port names you declared when registering the block.

Here is a minimal block function that builds a one-variable model. It reads a
capacity value from the inputs, creates a variable bounded by that capacity,
and minimizes it.

```python
def build_supply(ctx):
    import arco
    cap = ctx.inputs["capacity"]
    model = arco.Model()
    x = model.add_variable(
        bounds=arco.Bounds(lower=0.0, upper=cap),
        name="supply",
    )
    model.minimize(expr=x)
    return model
```

The function must return an `arco.Model` with an objective set. Arco will call
`model.solve()` on that model and collect the result.

## Registering blocks on a model

You register blocks with `model.add_block()`. This method takes the build
callable, a unique name, and optional `inputs` and `outputs` dictionaries that
declare the block's ports. It returns a `BlockHandle` whose `.input(key)` and
`.output(key)` methods produce port references for linking.

The `inputs` dictionary serves double duty: it declares the port names and
provides the initial data values. For root blocks that receive no linked
outputs, pass the actual values you want the block to use. For downstream
blocks whose inputs come entirely from links, you can omit the `inputs`
parameter.

The `extract` parameter is a callable that receives the `SolveResult` and the
`BlockContext` after the solve, and returns a dictionary of output values. These
values become available on the block's output ports.

Let's register two blocks on a model and link them. The first block solves a
supply problem with a capacity of 50 and outputs the optimal supply level. The
second block reads that supply level as an input and uses it as a demand floor.

```python doctest
>>> import arco
>>>
>>> def build_supply(ctx):
...     cap = ctx.inputs["capacity"]
...     m = arco.Model()
...     x = m.add_variable(bounds=arco.Bounds(lower=0.0, upper=cap), name="supply")
...     m.minimize(expr=x)
...     return m
>>>
>>> def extract_supply(solution, ctx):
...     return {"level": solution.get_primal(index=0)}
>>>
>>> def build_demand(ctx):
...     floor = ctx.inputs["supply_level"]
...     m = arco.Model()
...     y = m.add_variable(bounds=arco.Bounds(lower=floor, upper=100.0), name="demand")
...     m.minimize(expr=y)
...     return m
>>>
>>> model = arco.Model()
>>> h1 = model.add_block(
...     build_supply,
...     name="supply",
...     inputs={"capacity": 50.0},
...     outputs={"level": None},
...     extract=extract_supply,
... )
>>> h2 = model.add_block(
...     build_demand,
...     name="demand",
... )
>>> model.link(h1.output("level"), h2.input("supply_level"))
>>> model.has_blocks
True
```

At this point no optimization has run. The model knows about two blocks and one
link between them, but `solve()` has not been called yet.

## Solving a composed model

When you call `solve()` on a model that contains blocks, arco builds a
dependency graph from the links, determines the execution order, and solves
each block in sequence. The result object has a `.blocks` attribute that gives
you access to the per-block `SolveResult`.

```python doctest
>>> import arco
>>>
>>> def build_supply(ctx):
...     cap = ctx.inputs["capacity"]
...     m = arco.Model()
...     x = m.add_variable(bounds=arco.Bounds(lower=0.0, upper=cap), name="supply")
...     m.minimize(expr=x)
...     return m
>>>
>>> def extract_supply(solution, ctx):
...     return {"level": solution.get_primal(index=0)}
>>>
>>> def build_demand(ctx):
...     floor = ctx.inputs["supply_level"]
...     m = arco.Model()
...     y = m.add_variable(bounds=arco.Bounds(lower=floor, upper=100.0), name="demand")
...     m.minimize(expr=y)
...     return m
>>>
>>> model = arco.Model()
>>> h1 = model.add_block(
...     build_supply,
...     name="supply",
...     inputs={"capacity": 50.0},
...     outputs={"level": None},
...     extract=extract_supply,
... )
>>> h2 = model.add_block(
...     build_demand,
...     name="demand",
... )
>>> model.link(h1.output("level"), h2.input("supply_level"))
>>> result = model.solve(log_to_console=False)
>>> result.blocks.keys()
['supply', 'demand']
>>> result.blocks["supply"].is_optimal()
True
>>> result.blocks["demand"].is_optimal()
True
```

The supply block minimizes over a capacity of 50. Because the objective
minimizes the supply variable and there are no other constraints, the optimal
supply is 0. That value is extracted and passed to the demand block, which uses
it as the lower bound for its own variable. Each block solves independently and
the results are stitched together.

> [!NOTE]
> The `result.blocks` object behaves like a read-only dictionary. You can call
> `.keys()`, `.values()`, and `.items()` to iterate over the per-block results,
> or use bracket notation to access a specific block by name.

## The BlockModel API

For more control over block orchestration, arco provides `BlockModel` in the
`arco.blocks` submodule. A `BlockModel` manages its own collection of blocks
and links, and its `solve()` method returns a list of `BlockRun` objects with
detailed diagnostics.

```python
from arco.blocks import BlockModel

bm = BlockModel(name="pipeline")
b1 = bm.add_block(build_supply, name="supply", inputs={"capacity": 50.0}, outputs={"level": None}, extract=extract_supply)
b2 = bm.add_block(build_demand, name="demand", inputs_schema={"supply_level": None})
bm.link(b1.output("level"), b2.input("supply_level"))
runs = bm.solve()
```

Each `BlockRun` carries the block name, the solution, extracted outputs, any
attachments stored via `ctx.attach()`, and a `diagnostics` object with build
and solve timing in milliseconds.

## Next steps

You now know how to split a problem into blocks, connect them with typed ports,
and solve the composed model. The [Building Optimization Models](../how-to/building-optimization-models.md)
guide covers additional patterns for structuring your models.
