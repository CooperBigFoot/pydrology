# Common Mistakes When Adding a New Model to PyDrology

> **Audience:** LLM coding agents. Each entry is self-contained with error symptom, cause, and fix.

## Quick Reference Table

| ID  | Severity      | Error / Symptom                                              | One-line Fix                                                       |
|-----|---------------|--------------------------------------------------------------|--------------------------------------------------------------------|
| 1.1 | `SILENT-BUG`  | Wrong values in output arrays                                | Match flux field order across all 5 locations                      |
| 1.2 | `SILENT-BUG`  | State reconstructs with wrong values                         | Match state serialization order across 3 locations                 |
| 1.3 | `SILENT-BUG`  | Calibration converges to wrong optimum                       | Match parameter array index order across 3 locations               |
| 1.4 | `WILL-CRASH`  | `ImportError: cannot import name '<model>_run'`              | Complete all 5 registration steps                                  |
| 1.5 | `WILL-CRASH`  | `KeyError: Unknown model '<model>'`                          | Add auto-registration block at bottom of Python shim               |
| 2.1 | `WILL-CRASH`  | `ValueError: array must be C-contiguous`                     | Use `np.ascontiguousarray()` not `np.asarray()`                    |
| 2.2 | `WILL-CRASH`  | `TypeError: missing required positional argument`            | Add `#[pyo3(signature = (...))]` with defaults                     |
| 2.3 | `CONFUSING`   | Python gets dicts but typed results exist                    | Both dict API and typed pyclass API coexist by design              |
| 2.4 | `WILL-CRASH`  | `error[E0609]: no field 'X' on type FluxesTimeseries`        | Macro field names must match Fluxes struct exactly                 |
| 3.1 | `WILL-CRASH`  | `Fluxes derive: all fields must be f64`                      | All Fluxes struct fields must be `f64`                             |
| 3.2 | `CONFUSING`   | Unclear when to use Context vs `()`                          | Use Context for precomputed data constant per run                  |
| 3.3 | `SILENT-BUG`  | `STATE_SIZE` doesn't match actual state array                | Count all elements including UH buffers                            |
| 3.4 | `CONFUSING`   | Duplicate boilerplate in FluxesTimeseriesOps                 | Impl block is required despite macro -- copy verbatim              |
| 4.1 | `WILL-CRASH`  | `ValueError: array must be C-contiguous` via `.astype()`     | `.astype()` may not preserve contiguity                            |
| 4.2 | `CONFUSING`   | Model not in `registry.list_models()` after import           | Auto-registration requires import chain from `__init__.py`         |
| 4.3 | `WILL-CRASH`  | `State.from_array()` fails for variable-size models          | Pass `n_zones` or `n_layers` argument                              |
| 4.4 | `SILENT-BUG`  | Registry validates exports but not field order               | Registry only checks presence, not correctness                     |

---

## Section 1: Cross-Layer Gotchas

### 1.1 Flux Field Order Mismatch `[SILENT-BUG]`

**Symptom:** Output arrays contain valid values but assigned to wrong variable names.

**Cause:** Field order in one of the 5 locations disagrees. The macros use positional mapping, not names.

**The 5 locations that MUST have identical field order (source of truth is #1):**
1. `crates/pydrology-core/src/<model>/fluxes.rs` -- Rust `Fluxes` struct
2. `crates/pydrology-python/src/<model>.rs` -- `define_timeseries_result!` field list
3. `crates/pydrology-python/src/<model>.rs` -- `define_step_result!` field list
4. `crates/pydrology-python/src/<model>.rs` -- `timeseries_to_dict!` / `fluxes_to_dict!` field list
5. `python/pydrology/models/<model>.py` -- Python `*Fluxes` dataclass field order

**Wrong:** Fields in different order between Rust and Python:
```python
# Python shim -- REVERSED from Rust
class MyModelFluxes:
    streamflow: np.ndarray   # WRONG: this is index 0 in Rust = actual_et
    actual_et: np.ndarray
```
**Right:** Copy field list from Rust `Fluxes` struct verbatim into all 4 other locations.

---

### 1.2 State Serialization Order `[SILENT-BUG]`

**Symptom:** State roundtrips via `to_vec()`/`from_slice()` silently produce corrupt state.

**Cause:** Array index layout in `to_vec()` doesn't match `from_slice()` or Python `__array__()`.

**The 3 locations that MUST match:**
1. Rust `State::to_vec()` in `crates/pydrology-core/src/<model>/state.rs`
2. Rust `State::from_slice()` in the same file
3. Python `State.__array__()` and `State.from_array()` in `python/pydrology/models/<model>.py`

**Wrong:**
```python
def __array__(self, dtype=None):
    arr[0] = self.routing_store      # WRONG: Rust has production_store at index 0
    arr[1] = self.production_store
```
**Right:** Copy index layout from Rust `to_vec()`. Rust is the source of truth.

---

### 1.3 Parameter Array Index Order `[SILENT-BUG]`

**Symptom:** Calibration converges but model output is wrong.

**Cause:** `from_array()` index mapping doesn't match `to_array()` or `PARAM_NAMES`.

**3 locations that MUST match:** Rust `to_array()`, Rust `from_array()`, Python `__array__()`/`from_array()`.

**Wrong:**
```python
def from_array(cls, arr):
    return cls(x1=float(arr[0]), x2=float(arr[2]), x3=float(arr[1]), ...)  # swapped
```
**Right:** Indices must match `PARAM_NAMES` order: `arr[0]`=x1, `arr[1]`=x2, etc.

---

### 1.4 Five-Step Registration Chain `[WILL-CRASH]`

**Symptom:** `ImportError: cannot import name '<model>_run' from 'pydrology._core.<model>'`

**Cause:** Missed one of 5 steps. All must be done:

1. `mod <model>;` in `crates/pydrology-python/src/lib.rs`
2. `<model>::register(m)?;` in `_core()` function in the same file
3. `"<model>"` added to `sys.modules` loop array in the same file:
   ```rust
   for name in &["cemaneige", "gr2m", "gr6j", "hbv_light", "<model>", "metrics"] {
   ```
4. `pub fn register(parent: &Bound<'_, PyModule>)` in `crates/pydrology-python/src/<model>.rs`:
   ```rust
   pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
       let m = PyModule::new(parent.py(), "<model>")?;
       m.add_function(wrap_pyfunction!(<model>_run, &m)?)?;
       m.add_function(wrap_pyfunction!(<model>_step, &m)?)?;
       m.add_class::<MyModelResult>()?;
       m.add_class::<MyModelStepFluxes>()?;
       parent.add_submodule(&m)?;
       Ok(())
   }
   ```
5. `import pydrology.models.<model>` in `python/pydrology/__init__.py`

---

### 1.5 Missing Auto-Registration Block `[WILL-CRASH]`

**Symptom:** `KeyError: Unknown model '<model>'. Available models: gr2m, gr6j, gr6j_cemaneige, hbv_light`

**Cause:** Python model file is missing the registration block at the bottom.

**Wrong:** File ends after `__all__` with no registration code.

**Right:** Add at the very bottom of `python/pydrology/models/<model>.py`:
```python
# Auto-register
import pydrology.models.<model> as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("<model>", _self)
```

---

## Section 2: PyO3 Binding Gotchas

### 2.1 Array Contiguity at PyO3 Boundary `[WILL-CRASH]`

**Symptom:** `ValueError: array must be C-contiguous`

**Cause:** Numpy array passed to Rust is not C-contiguous (slice, transpose, or Fortran-ordered).

**Wrong:**
```python
state_arr = np.asarray(state, dtype=np.float64)  # does NOT guarantee contiguity
```
**Right:**
```python
state_arr = np.ascontiguousarray(state, dtype=np.float64)  # guarantees C-contiguous
```

`np.asarray()` preserves existing memory layout. `np.ascontiguousarray()` copies if needed, which `PyReadonlyArray1::as_slice()` requires.

---

### 2.2 Missing pyo3 Signature Attribute `[WILL-CRASH]`

**Symptom:** `TypeError: <model>_run() missing 1 required positional argument: 'initial_state'`

**Cause:** Without `#[pyo3(signature = (...))]`, PyO3 treats `Option<T>` as required.

**Wrong:**
```rust
#[pyfunction]
fn mymodel_run<'py>(py: Python<'py>, params: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,  // no signature attr
) -> PyResult<Bound<'py, PyDict>> { ... }
```
**Right:**
```rust
#[pyfunction]
#[pyo3(signature = (params, precip, pet, initial_state=None))]
fn mymodel_run<'py>(py: Python<'py>, params: PyReadonlyArray1<'py, f64>,
    precip: PyReadonlyArray1<'py, f64>, pet: PyReadonlyArray1<'py, f64>,
    initial_state: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> { ... }
```

---

### 2.3 Dict vs Typed Results Confusion `[CONFUSING]`

**Symptom:** Both `MyModelResult` pyclass and dict-returning `mymodel_run()` exist.

**Rule:** Two parallel APIs coexist by design:
- **Dict API** (`timeseries_to_dict!`): Primary. Called by Python shims. Returns `PyDict`.
- **Typed API** (`define_timeseries_result!`): Frozen `#[pyclass]` registered but not used by shims.

Implement both. The Python shim calls the dict-returning function at runtime.

---

### 2.4 Macro Field Name Mismatch `[WILL-CRASH]`

**Symptom:** `error[E0609]: no field 'evapotranspiration' on type 'FluxesTimeseries'`

**Cause:** Field name in macro doesn't exactly match the core `Fluxes` struct.

**Wrong:**
```rust
define_timeseries_result! {
    pub struct MyModelResult from pydrology_core::mymodel::fluxes::FluxesTimeseries {
        evapotranspiration,  // WRONG: Fluxes has `actual_et`
    }
}
```
**Right:** Copy field names exactly from the Rust `Fluxes` struct -- use `actual_et`.

---

## Section 3: Rust Core Gotchas

### 3.1 Non-f64 Fluxes Fields `[WILL-CRASH]`

**Symptom:** `Fluxes derive: all fields must be f64`

**Cause:** `#[derive(Fluxes)]` enforces all fields are `pub <name>: f64`.

**Wrong:**
```rust
pub struct Fluxes {
    pub streamflow: f64,
    pub count: usize,    // WRONG
}
```
**Right:** All fields `f64`. Cast at use site if needed.

---

### 3.2 When to Use Context vs `()` `[CONFUSING]`

**Symptom:** Unclear whether to define a Context struct or use `type Context = ()`.

**Rule:** Use Context for data derived from parameters, constant per run, expensive per timestep.

| Model     | Context        | Reason                                        |
|-----------|----------------|-----------------------------------------------|
| GR6J      | `GR6JContext`  | UH ordinates precomputed from `x4`            |
| HBV-Light | `HBVContext`   | Triangular UH weights from `maxbas`           |
| GR2M      | `()`           | No precomputed data needed                    |

```rust
// With Context:
type Context = MyModelContext;
fn prepare(params: &Self::Params) -> Self::Context {
    MyModelContext { uh_weights: compute_weights(params.maxbas) }
}
// Without Context:
type Context = ();
fn prepare(_params: &Self::Params) -> Self::Context {}
```

---

### 3.3 STATE_SIZE Mismatch `[SILENT-BUG]`

**Symptom:** `from_slice()` returns `Err("expected N state elements, got M")` or corrupt state.

**Cause:** `STATE_SIZE` constant doesn't match actual serialized element count.

**Wrong:**
```rust
pub const STATE_SIZE: usize = 3;  // Forgot UH buffers
// But to_array() writes 3 + 20 + 40 = 63 elements
```
**Right:**
```rust
pub const STATE_SIZE: usize = 63;  // 3 scalars + 20 UH1 + 40 UH2
```

**Formula:** `STATE_SIZE` = scalar fields + sum of all array lengths. GR6J: 63. GR2M: 2. HBV-Light (1 zone): 12. Python `STATE_SIZE` must match.

---

### 3.4 FluxesTimeseriesOps Boilerplate `[CONFUSING]`

**Symptom:** Why write this impl when `#[derive(Fluxes)]` generates the methods?

**Cause:** The macro generates inherent methods on `FluxesTimeseries`. The `FluxesTimeseriesOps<Fluxes>` trait needs an explicit impl to bridge them. `HydrologicalModel::run()` calls trait methods.

**Rule:** ALWAYS add this in `fluxes.rs` (change only type names):
```rust
impl FluxesTimeseriesOps<Fluxes> for FluxesTimeseries {
    fn with_capacity(n: usize) -> Self { FluxesTimeseries::with_capacity(n) }
    fn push(&mut self, f: &Fluxes) { FluxesTimeseries::push(self, f); }
    fn len(&self) -> usize { FluxesTimeseries::len(self) }
    fn is_empty(&self) -> bool { FluxesTimeseries::is_empty(self) }
    fn with_len(n: usize) -> Self { FluxesTimeseries::with_len(n) }
    unsafe fn write_unchecked(&mut self, t: usize, f: &Fluxes) {
        unsafe { FluxesTimeseries::write_unchecked(self, t, f); }
    }
}
```

---

## Section 4: Python Layer Gotchas

### 4.1 `.astype()` Breaks Contiguity `[WILL-CRASH]`

**Symptom:** `ValueError: array must be C-contiguous` (from `crates/pydrology-python/src/convert.rs`)

**Cause:** `.astype(np.float64)` does not guarantee contiguity if the input was non-contiguous.

**Wrong:**
```python
forcing.precip.astype(np.float64)  # RISKY: not guaranteed contiguous
```
**Right:**
```python
np.ascontiguousarray(forcing.precip, dtype=np.float64)  # SAFE
```

The existing GR6J shim uses `.astype()` because `ForcingData` guarantees contiguous arrays. For new models, prefer `np.ascontiguousarray()`.

---

### 4.2 Import-Triggered Registration `[CONFUSING]`

**Symptom:** `registry.list_models()` doesn't include your model.

**Cause:** Registration requires a complete import chain:
1. `python/pydrology/__init__.py` has `import pydrology.models.<model>`
2. This executes the model file top to bottom
3. `register("<model>", _self)` at the bottom runs
4. Model enters global `_models` dict

Miss step 1: file never imported. Miss step 3: file imported but nothing registers.

---

### 4.3 Variable-Size State `from_array()` `[WILL-CRASH]`

**Symptom:** `IndexError` or wrong state reconstruction.

**Cause:** Models with variable-size state (like HBV-Light) need zone count to partition the array.

**Wrong:**
```python
new_state = State.from_array(state_arr)  # Doesn't know n_zones
```
**Right:**
```python
new_state = State.from_array(state_arr, n_zones=state.n_zones)
```

Also affects `_make_initial_state()` and `_reconstruct_state()` in conformance tests.

---

### 4.4 Registry Validates Exports, Not Correctness `[SILENT-BUG]`

**Symptom:** Model registers fine but produces wrong results.

**What registry checks:** `PARAM_NAMES`, `DEFAULT_BOUNDS`, `STATE_SIZE`, `SUPPORTED_RESOLUTIONS` exist; `Parameters`/`State` have `from_array()`/`__array__()`; `run`/`step` exist.

**What registry does NOT check:** Field order correctness (1.1, 1.2, 1.3), `STATE_SIZE` vs actual array length (3.3), `PARAM_NAMES` vs actual fields, bound reasonableness, flux name consistency.

**Rule:** Registry validation is necessary but not sufficient. Always run `uv run pytest tests/` to catch ordering bugs.
