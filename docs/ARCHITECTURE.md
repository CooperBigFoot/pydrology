# Architecture

PyDrology's numerical core is written in Rust and exposed to Python through [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/). This document describes the internal architecture for contributors.

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Python API layer  (python/pydrology/)              │
│  Parameters · State · ForcingData · ModelOutput      │
│  Registry · Calibration · Utilities                  │
├─────────────────────────────────────────────────────┤
│  PyO3 binding layer  (crates/pydrology-python/)     │
│  Array validation · Type conversion · Submodules     │
├─────────────────────────────────────────────────────┤
│  Rust core  (crates/pydrology-core/)                │
│  GR2M · GR6J · HBV-Light · CemaNeige               │
│  Traits · Processes · Unit hydrographs               │
├─────────────────────────────────────────────────────┤
│  Proc macro  (crates/pydrology-macros/)             │
│  #[derive(Fluxes)] → FluxesTimeseries generation    │
└─────────────────────────────────────────────────────┘
```

The Cargo workspace contains three crates:

| Crate | Type | Role |
|-------|------|------|
| `pydrology-core` | rlib | Pure Rust models. No PyO3 dependency. |
| `pydrology-python` | cdylib | PyO3/numpy bindings. Thin translation layer. |
| `pydrology-macros` | proc-macro | `#[derive(Fluxes)]` code generation. |

## Data Flow: A `run()` Call

Using GR6J as the canonical example, here is what happens when Python calls `gr6j.run(params, forcing)`:

```
Python                          PyO3 boundary                    Rust core
──────                          ──────────────                   ─────────
gr6j.run(params, forcing)
  │
  ├─ np.ascontiguousarray(params)
  ├─ np.ascontiguousarray(state)
  │
  └─► _rust.gr6j_run(params_arr, precip, pet, state_arr)
        │
        ├─ checked_slice(&params, 6, "params")     ← validates length + C-contiguity
        ├─ contiguous_slice(&precip)                ← validates C-contiguity
        ├─ Parameters::new(p[0]..p[5])              ← construct Rust params
        ├─ State::from_array(&arr)                  ← construct Rust state
        │
        └─► run::run(&params, precip, pet, state)
              │
              ├─ prepare(&params)                   ← precompute UH ordinates (Context)
              ├─ for each timestep:
              │    step(&state, &params, &forcing, &context)
              │    write_unchecked(t, &fluxes)       ← indexed write to FluxesTimeseries
              │
              └─► FluxesTimeseries { pet: Vec, precip: Vec, ... }
                    │
        ◄───────────┘
        │
        ├─ timeseries_to_dict!(py, result, pet, precip, ...)
        │   └─ PyArray1::from_vec() for each field  ← zero-copy where possible
        │
        └─► PyDict { "pet": ndarray, "precip": ndarray, ... }
              │
  ◄───────────┘
  │
  ├─ GR6JFluxes(**result)                           ← wrap in frozen dataclass
  └─► ModelOutput(time, fluxes, snow=None)
```

## Trait System

### `HydrologicalModel`

The central trait in `pydrology-core`. Each single-zone model implements it:

```rust
pub trait HydrologicalModel {
    const NAME: &'static str;
    type Params;
    type State: Clone;
    type Forcing: Copy;
    type Fluxes;
    type FluxesTimeseries: FluxesTimeseriesOps<Self::Fluxes>;
    type Context;  // precomputed data, e.g. UH ordinates

    fn prepare(params: &Self::Params) -> Self::Context;
    fn initialize_state(params: &Self::Params) -> Self::State;
    fn step(state: &Self::State, params: &Self::Params,
            forcing: &Self::Forcing, context: &Self::Context)
        -> (Self::State, Self::Fluxes);
    fn run(params: &Self::Params, forcing: &[Self::Forcing],
           initial_state: Option<&Self::State>)
        -> Self::FluxesTimeseries;  // default impl loops over step()
}
```

The default `run()` implementation calls `prepare()` once, then loops over `step()`, writing results via `unsafe write_unchecked()` for zero-overhead indexed writes.

### `ModelParams` and `ModelState`

Supporting traits for uniform serialization:

- **`ModelParams`**: `from_array(&[f64])`, `to_array() -> Vec<f64>`, plus compile-time constants (`N_PARAMS`, `PARAM_NAMES`, `PARAM_BOUNDS`).
- **`ModelState`**: `from_slice(&[f64])`, `to_vec() -> Vec<f64>`, `array_len()`.

### `FluxesTimeseriesOps<F>`

Trait for the timeseries collection type, auto-implemented by the `#[derive(Fluxes)]` macro:

```rust
pub trait FluxesTimeseriesOps<F> {
    fn with_capacity(n: usize) -> Self;
    fn push(&mut self, f: &F);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn with_len(n: usize) -> Self;
    unsafe fn write_unchecked(&mut self, t: usize, f: &F);
}
```

### When to Use the Trait vs. Standalone Functions

The `HydrologicalModel` trait works for single-zone models with a fixed parameter set (GR2M, GR6J, HBV-Light single-zone). Models requiring extra runtime arguments — like multi-zone HBV-Light (zone elevations, fractions) or the coupled GR6J-CemaNeige model — are implemented as standalone `run()` / `step()` functions in the core crate.

## Proc Macro: `#[derive(Fluxes)]`

Applied to a struct with all `f64` fields, it generates:

1. A companion `FluxesTimeseries` struct where each field is `Vec<f64>`
2. Methods: `with_capacity(n)`, `push(&Fluxes)`, `len()`, `is_empty()`, `with_len(n)`, `write_unchecked(t, &Fluxes)`
3. A `field_names() -> &[&str]` associated function on the original struct

Example:

```rust
#[derive(Fluxes)]
pub struct Fluxes {
    pub streamflow: f64,
    pub actual_et: f64,
}
// Generates: FluxesTimeseries { streamflow: Vec<f64>, actual_et: Vec<f64> }
```

Custom naming: `#[fluxes(timeseries_name = "MyTimeseries")]`.

Constraints: all fields must be `f64`, struct must have named fields.

## PyO3 Boundary Design

### What Crosses the Boundary

| Direction | Format | Notes |
|-----------|--------|-------|
| Python → Rust (params, state) | `numpy.ndarray` (f64, C-contiguous) | Validated by `convert.rs` |
| Python → Rust (forcing) | `numpy.ndarray` (f64, C-contiguous) | One array per variable |
| Rust → Python (timeseries) | `PyDict` of `numpy.ndarray` | Via `timeseries_to_dict!` macro |
| Rust → Python (step fluxes) | `PyDict` of `f64` | Via `fluxes_to_dict!` macro |

State is passed as flat numpy arrays at the boundary — **not** as `#[pyclass]` objects. This keeps the Python layer free to define its own dataclass structure while the Rust side uses its own struct layout.

### Array Validation (`convert.rs`)

Four validation functions ensure arrays meet Rust expectations:

| Function | Validates |
|----------|-----------|
| `contiguous_slice(arr)` | C-contiguity only |
| `checked_slice(arr, len, name)` | C-contiguity + exact length |
| `optional_checked_slice(arr, len, name)` | Same as above, but `Option<>` |
| `checked_slice_min(arr, min_len, name)` | C-contiguity + minimum length |

### Module Registration

PyO3 submodules must be registered in `sys.modules` for `from pydrology._core.gr6j import gr6j_run` to work:

```rust
fn register_submodule(py: Python, parent_name: &str, child: &Bound<PyModule>) -> PyResult<()> {
    let full_name = format!("{}.{}", parent_name, child.name()?);
    py.import("sys")?.getattr("modules")?.set_item(full_name, child)?;
    Ok(())
}
```

### Declarative Macros (`macros.rs`)

| Macro | Input | Output |
|-------|-------|--------|
| `define_timeseries_result!` | Core `FluxesTimeseries` | Frozen `#[pyclass]` with `Py<PyArray1<f64>>` fields |
| `define_step_result!` | Core `Fluxes` | Frozen `#[pyclass]` with `f64` fields |
| `timeseries_to_dict!` | `FluxesTimeseries` + field list | `PyDict` of numpy arrays |
| `fluxes_to_dict!` | `Fluxes` + field list | `PyDict` of f64 scalars |

## Python Layer

### Shim Pattern

Each model is a single Python file (`python/pydrology/models/<model>.py`) that:

1. Defines `Parameters` (frozen dataclass with `__array__` and `from_array`)
2. Defines `State` (mutable dataclass with `__array__`, `from_array`, `initialize`)
3. Defines a `*Fluxes` frozen dataclass for outputs
4. Exports `run()` and `step()` that delegate to Rust:

```python
def step(state, params, precip, pet):
    from pydrology._core import gr6j as _rust
    state_arr = np.ascontiguousarray(state, dtype=np.float64)
    params_arr = np.ascontiguousarray(params, dtype=np.float64)
    new_state_arr, fluxes = _rust.gr6j_step(state_arr, params_arr, precip, pet, ...)
    return State.from_array(new_state_arr), fluxes
```

5. Auto-registers with the model registry at import time

### Model Registry

The registry (`pydrology/registry.py`) validates that each model exports the required contract symbols (`PARAM_NAMES`, `DEFAULT_BOUNDS`, `STATE_SIZE`, `SUPPORTED_RESOLUTIONS`, `Parameters`, `State`, `run`, `step`).

### Calibration Integration

The calibration module (`pydrology/calibration/`) uses `Parameters.from_array()` / `__array__()` to convert between optimizer vectors and typed parameter objects. The registry provides bounds via `DEFAULT_BOUNDS`.

### `ModelOutput[F]`

Generic wrapper in `pydrology/outputs.py`. `F` is the model-specific fluxes dataclass (e.g., `GR6JFluxes`). Provides `.streamflow`, `.to_dataframe()`, and optional `.snow` / `.snow_layers` for snow-enabled models.

## Adding a New Model

> **Detailed contributor guides:**
> - Full tutorial with code templates: [`docs/ADDING_A_MODEL.md`](ADDING_A_MODEL.md)
> - Step-by-step checklist: [`docs/MODEL_CHECKLIST.md`](MODEL_CHECKLIST.md)
> - Common mistakes and fixes: [`docs/COMMON_MISTAKES.md`](COMMON_MISTAKES.md)

### 1. Rust Core (`crates/pydrology-core/src/<model>/`)

Create the following files:

- `mod.rs` — module declarations
- `constants.rs` — physical and numerical constants
- `params.rs` — implement `ModelParams` trait
- `state.rs` — implement `ModelState` trait
- `processes.rs` — individual process functions (production, routing, etc.)
- `fluxes.rs` — define `Fluxes` struct with `#[derive(Fluxes)]`
- `run.rs` — implement `HydrologicalModel` trait (or standalone `run`/`step`)

Add `pub mod <model>;` to `crates/pydrology-core/src/lib.rs`.

### 2. PyO3 Bindings (`crates/pydrology-python/src/<model>.rs`)

- Use `define_timeseries_result!` and `define_step_result!` for typed results
- Write `<model>_run()` and `<model>_step()` `#[pyfunction]`s
- Create a `register(m: &Bound<PyModule>)` function
- Add to `lib.rs`: module import, `register()` call, and `sys.modules` entry

### 3. Python API (`python/pydrology/models/<model>.py`)

- Define `Parameters`, `State`, `*Fluxes` dataclasses
- Write `run()` and `step()` shims that call `_rust.<model>_run()` / `_rust.<model>_step()`
- Add auto-registration at module bottom
- Import in `python/pydrology/__init__.py` to trigger registration

## Key Design Decisions

### State as Arrays, Not PyClasses

State crosses the Python/Rust boundary as flat `numpy.ndarray` rather than `#[pyclass]` objects. This allows the Python side to freely restructure state (e.g., separate `production_store` vs. `uh1_states` fields) without changing the Rust interface. The Rust side works with fixed-size arrays internally.

### Frozen Result Classes

All `#[pyclass]` result types in the binding layer use `frozen` mode. This prevents mutation after construction, which is important for thread safety if results are shared across threads, and matches the semantic that simulation outputs are immutable records.

### `unsafe write_unchecked`

The default `HydrologicalModel::run()` implementation pre-allocates `FluxesTimeseries` with `with_len(n)` and writes via `unsafe write_unchecked(t, &fluxes)` instead of `push()`. This avoids bounds checking in the inner loop and eliminates the overhead of `Vec::push` capacity checks. Safety is guaranteed by the loop invariant `t < n`.

### Coupled Models Bypass the Trait

The GR6J-CemaNeige coupled model and HBV-Light multi-zone model take additional runtime arguments (layer elevations, fractions, gradients, mean annual solid precipitation) that don't fit the `HydrologicalModel` trait signature. These are implemented as standalone functions in the core crate, with corresponding standalone PyO3 bindings.
