# New Model Checklist

> Target audience: LLM coding agents. No prose. Every step is actionable.

---

## 1. Decision Tree

Answer all 5 questions. Collect the labels.

| # | Question | IF YES | IF NO |
|---|----------|--------|-------|
| 1 | Does the model have a fixed parameter set with no extra runtime arguments? | Implement `HydrologicalModel` trait -> `TRAIT` | Standalone `run()`/`step()` functions -> `STANDALONE` |
| 2 | Does the model need precomputed data derived from parameters? (e.g., UH ordinates) | Define a Context struct -> `HAS_CONTEXT` | Use `type Context = ()` -> `NO_CONTEXT` |
| 3 | Is the state size fixed (same for all configurations)? | Use `#[derive(Copy)]` on State, fixed `STATE_SIZE` -> `FIXED_STATE` | Use `compute_state_size()` function -> `VARIABLE_STATE` |
| 4 | Does the model require temperature forcing? | Add `temp` field to forcing + Python shim -> `NEEDS_TEMP` | Only `precip` and `pet` -> `NO_TEMP` |
| 5 | Is this a coupled model (e.g., snow + rainfall-runoff)? | May need two sets of state/fluxes -> `COUPLED` | Single state + single fluxes -> `SINGLE` |

**Result:** comma-separated labels (e.g., `TRAIT, NO_CONTEXT, FIXED_STATE, NO_TEMP, SINGLE`)

---

## 2. Phase-by-Phase Checklist

Replace `<model>` with the lowercase model name (e.g., `gr2m`, `hbv_light`).

### Phase 1: Rust Core

- [ ] **1.1** Create `crates/pydrology-core/src/<model>/constants.rs` -- Define `PARAM_NAMES`, `PARAM_BOUNDS`, `STATE_SIZE`, `SUPPORTED_RESOLUTIONS`, numerical safeguard constants [blocks: 1.3, 1.4, 1.6]
- [ ] **1.2** Create `crates/pydrology-core/src/<model>/mod.rs` -- Declare submodules: `constants`, `fluxes`, `params`, `processes`, `run`, `state` [blocks: 1.8]
- [ ] **1.3** Create `crates/pydrology-core/src/<model>/params.rs` -- Define `Parameters` struct, impl `ModelParams` with `from_array`/`to_array` [blocked by: 1.1] [blocks: 1.4, 1.6, 1.7]
- [ ] **1.4** Create `crates/pydrology-core/src/<model>/state.rs` -- Define `State` struct, impl `ModelState` with `to_vec`/`from_slice`/`array_len` [blocked by: 1.1, 1.3] [blocks: 1.6, 1.7]
- [ ] **1.5** Create `crates/pydrology-core/src/<model>/fluxes.rs` -- Define `Fluxes` with `#[derive(Fluxes)]`, impl `FluxesTimeseriesOps<Fluxes>` for `FluxesTimeseries` [blocks: 1.7, 2.1]
- [ ] **1.6** Create `crates/pydrology-core/src/<model>/processes.rs` -- Pure math functions for each model process [blocked by: 1.1, 1.3, 1.4] [blocks: 1.7]
- [ ] **1.7** Create `crates/pydrology-core/src/<model>/run.rs` -- `step()` function, `run()` function, `HydrologicalModel` impl on marker struct [blocked by: 1.3, 1.4, 1.5, 1.6] [only if: `TRAIT`]
- [ ] **1.7a** Create `crates/pydrology-core/src/<model>/run.rs` -- Standalone `step()` and `run()` functions (no trait impl) [blocked by: 1.3, 1.4, 1.5, 1.6] [only if: `STANDALONE`]
- [ ] **1.8** Edit `crates/pydrology-core/src/lib.rs` -- Add `pub mod <model>;` in alphabetical order [blocked by: 1.2]
- [ ] **1.9** Verify: `cargo test --workspace` -- All existing + new Rust tests pass, 0 failures [blocked by: 1.1-1.8]

### Phase 2: PyO3 Bindings

- [ ] **2.1** Create `crates/pydrology-python/src/<model>.rs` -- `define_timeseries_result!` struct, `define_step_result!` struct, `<model>_run()` pyfunction, `<model>_step()` pyfunction, `pub fn register()` [blocked by: 1.5, 1.7]
- [ ] **2.2** Edit `crates/pydrology-python/src/lib.rs` -- Add `mod <model>;` declaration, call `<model>::register(m)?;`, add `"<model>"` to `sys.modules` registration loop [blocked by: 2.1]
- [ ] **2.3** Verify: `cargo test --workspace` -- All tests still pass [blocked by: 2.2]

### Phase 3: Python Shim

- [ ] **3.1** Create `python/pydrology/models/<model>.py` -- Constants (`PARAM_NAMES`, `DEFAULT_BOUNDS`, `STATE_SIZE`, `SUPPORTED_RESOLUTIONS`), `Parameters` frozen dataclass, `State` dataclass, `Fluxes` frozen dataclass, `step()`, `run()`, `__all__`, auto-register via `pydrology.registry.register()` [blocked by: 2.2]
- [ ] **3.2** Edit `python/pydrology/__init__.py` -- Add `import pydrology.models.<model>  # noqa: F401 - triggers auto-registration` [blocked by: 3.1]
- [ ] **3.3** Build: `uv run maturin develop` -- Build succeeds with no errors [blocked by: 2.2]
- [ ] **3.4** Verify: `uv run python -c "from pydrology._core.<model> import <model>_run"` -- No ImportError [blocked by: 3.3]
- [ ] **3.5** Verify: `uv run python -c "from pydrology.models.<model> import run, step, Parameters, State"` -- All imports succeed [blocked by: 3.1, 3.3]
- [ ] **3.6** Verify: `uv run python -c "from pydrology import registry; assert '<model>' in registry.list_models()"` -- Model is registered [blocked by: 3.2, 3.5]

### Phase 4: Testing

- [ ] **4.1** Create `tests/models/<model>/__init__.py` -- Empty file [blocked by: 3.1]
- [ ] **4.2** Create `tests/models/<model>/test_run.py` -- Integration tests: `step()` returns (State, dict), `run()` returns ModelOutput, streamflow length matches forcing, non-negative streamflow, finite outputs, custom initial state [blocked by: 3.5]
- [ ] **4.3** Create `tests/models/<model>/test_types.py` -- Unit tests: Parameters `from_array`/`__array__` roundtrip, State `from_array`/`__array__` roundtrip, Parameters rejects wrong-length array, State.initialize produces valid state [blocked by: 3.5]
- [ ] **4.4** Edit `tests/test_model_conformance.py` -- Add `"<model>"` to `ALL_MODELS` list, add elif branch in `_model_fixtures()`, add elif branch in `_make_initial_state()`, add elif branch in `_reconstruct_state()` [blocked by: 3.6]
- [ ] **4.5** Verify: `uv run python -m pytest tests/models/<model>/ -v` -- Model-specific tests pass [blocked by: 4.2, 4.3]
- [ ] **4.6** Verify: `uv run python -m pytest tests/test_model_conformance.py -v` -- Conformance suite passes for all models [blocked by: 4.4]
- [ ] **4.7** Verify: `uv run python -m pytest -x` -- Full test suite passes, 0 failures [blocked by: 4.5, 4.6]

---

## 3. Dependency Map

### Step Dependencies

| Step | Blocked By | Blocks |
|------|-----------|--------|
| 1.1  | --        | 1.3, 1.4, 1.6 |
| 1.2  | --        | 1.8 |
| 1.3  | 1.1       | 1.4, 1.6, 1.7 |
| 1.4  | 1.1, 1.3  | 1.6, 1.7 |
| 1.5  | --        | 1.7, 2.1 |
| 1.6  | 1.1, 1.3, 1.4 | 1.7 |
| 1.7  | 1.3, 1.4, 1.5, 1.6 | 1.9, 2.1 |
| 1.8  | 1.2       | 1.9 |
| 1.9  | 1.1-1.8   | 2.1 |
| 2.1  | 1.5, 1.7  | 2.2 |
| 2.2  | 2.1       | 2.3, 3.1, 3.3 |
| 2.3  | 2.2       | 3.1 |
| 3.1  | 2.2       | 3.2, 3.5, 4.1 |
| 3.2  | 3.1       | 3.6 |
| 3.3  | 2.2       | 3.4, 3.5 |
| 3.4  | 3.3       | -- |
| 3.5  | 3.1, 3.3  | 3.6, 4.2, 4.3 |
| 3.6  | 3.2, 3.5  | 4.4 |
| 4.1  | 3.1       | 4.5 |
| 4.2  | 3.5       | 4.5 |
| 4.3  | 3.5       | 4.5 |
| 4.4  | 3.6       | 4.6 |
| 4.5  | 4.1, 4.2, 4.3 | 4.7 |
| 4.6  | 4.4       | 4.7 |
| 4.7  | 4.5, 4.6  | -- |

### Parallelizable Groups

- **Group A (parallel, no deps):** 1.1, 1.2, 1.5
- **Group B (parallel, after 1.1):** 1.3
- **Group C (parallel, after 1.3):** 1.4, 1.6
- **Group D (sequential, after Group C + 1.5):** 1.7 -> 1.8 -> 1.9
- **Group E (sequential, after 1.9):** 2.1 -> 2.2 -> 2.3
- **Group F (parallel, after 2.2):** 3.1, 3.3
- **Group G (parallel, after 3.1 + 3.3):** 3.2, 3.4, 3.5, 4.1
- **Group H (parallel, after 3.5):** 3.6, 4.2, 4.3
- **Group I (sequential, after Group H):** 4.4 -> 4.5 -> 4.6 -> 4.7

---

## 4. Verification Protocol

Run in order. Every command must succeed before proceeding.

```bash
# 1. Rust compiles and all Rust tests pass
cargo test --workspace
# Expected: 0 failures

# 2. Clippy passes with no warnings
cargo clippy --workspace
# Expected: 0 warnings

# 3. Python extension builds
uv run maturin develop
# Expected: exit code 0

# 4. Rust-level Python binding import works
uv run python -c "from pydrology._core.<model> import <model>_run"
# Expected: no error

# 5. Rust-level step binding import works
uv run python -c "from pydrology._core.<model> import <model>_step"
# Expected: no error

# 6. Python shim run/step import works
uv run python -c "from pydrology.models.<model> import run, step"
# Expected: no error

# 7. Python shim types import works
uv run python -c "from pydrology.models.<model> import Parameters, State"
# Expected: no error

# 8. Model appears in registry
uv run python -c "from pydrology import registry; assert '<model>' in registry.list_models()"
# Expected: no error

# 9. Registry info is populated
uv run python -c "from pydrology import registry; info = registry.get_model_info('<model>'); print(info)"
# Expected: prints model info dict

# 10. Model-specific tests pass
uv run python -m pytest tests/models/<model>/ -v
# Expected: all pass

# 11. Conformance suite passes
uv run python -m pytest tests/test_model_conformance.py -v
# Expected: all pass

# 12. Registry tests pass
uv run python -m pytest tests/test_registry.py -v
# Expected: all pass

# 13. Array contiguity tests pass
uv run python -m pytest tests/models/test_array_contiguity.py -v
# Expected: all pass

# 14. Full test suite passes
uv run python -m pytest -x
# Expected: 0 failures
```

---

## 5. Quick-Reference Table

### File Inventory

| Layer | File | What Goes In It | Template Model |
|-------|------|----------------|----------------|
| Rust Core | `crates/pydrology-core/src/<model>/mod.rs` | `pub mod` declarations for all submodules | `gr2m` |
| Rust Core | `crates/pydrology-core/src/<model>/constants.rs` | `PARAM_NAMES`, `PARAM_BOUNDS`, `STATE_SIZE`, `SUPPORTED_RESOLUTIONS` | `gr2m` |
| Rust Core | `crates/pydrology-core/src/<model>/params.rs` | `Parameters` struct, `impl ModelParams` | `gr2m` |
| Rust Core | `crates/pydrology-core/src/<model>/state.rs` | `State` struct, `impl ModelState` | `gr2m` (fixed) / `hbv_light` (variable) |
| Rust Core | `crates/pydrology-core/src/<model>/fluxes.rs` | `Fluxes` with `#[derive(Fluxes)]`, `impl FluxesTimeseriesOps` | `gr2m` |
| Rust Core | `crates/pydrology-core/src/<model>/processes.rs` | Pure math process functions | `gr2m` |
| Rust Core | `crates/pydrology-core/src/<model>/run.rs` | `step()`, `run()`, `HydrologicalModel` impl | `gr2m` (no context) / `gr6j` (with context) |
| Rust Core | `crates/pydrology-core/src/lib.rs` | Add `pub mod <model>;` alphabetically | -- |
| PyO3 | `crates/pydrology-python/src/<model>.rs` | `define_timeseries_result!`, `define_step_result!`, `_run()`, `_step()`, `register()` | `gr2m` |
| PyO3 | `crates/pydrology-python/src/lib.rs` | `mod <model>;`, `register()` call, `sys.modules` entry | -- |
| Python | `python/pydrology/models/<model>.py` | Constants, `Parameters`/`State`/`Fluxes` dataclasses, `step()`, `run()`, `__all__`, auto-register | `gr2m` |
| Python | `python/pydrology/__init__.py` | `import pydrology.models.<model>` for registration trigger | -- |
| Tests | `tests/models/<model>/__init__.py` | Empty file | -- |
| Tests | `tests/models/<model>/test_run.py` | Integration tests for `step()` and `run()` | `gr2m` |
| Tests | `tests/models/<model>/test_types.py` | Roundtrip tests for `Parameters`, `State` | `gr2m` |
| Tests | `tests/test_model_conformance.py` | Add model to `ALL_MODELS`, `_model_fixtures`, `_make_initial_state`, `_reconstruct_state` | -- |

### Template Selection by Archetype

| Archetype Labels | Best Template |
|-----------------|---------------|
| `TRAIT, NO_CONTEXT, FIXED_STATE, NO_TEMP, SINGLE` | `gr2m` |
| `TRAIT, HAS_CONTEXT, FIXED_STATE, NO_TEMP, SINGLE` | `gr6j` |
| `TRAIT, HAS_CONTEXT, VARIABLE_STATE, NEEDS_TEMP, SINGLE` | `hbv_light` |
| `STANDALONE, HAS_CONTEXT, FIXED_STATE, NEEDS_TEMP, COUPLED` | `gr6j_cemaneige` |

### Key Imports to Copy

```rust
// constants.rs
use crate::forcing::Resolution;

// params.rs
use crate::traits::ModelParams;

// state.rs
use crate::traits::ModelState;

// fluxes.rs
use pydrology_macros::Fluxes;
use crate::traits::FluxesTimeseriesOps;

// run.rs
use crate::traits::HydrologicalModel;
```

```python
# models/<model>.py
from pydrology._core import <model> as _rust   # Rust backend
from pydrology.outputs import ModelOutput       # Return type for run()
from pydrology.types import ForcingData, Resolution
from pydrology.registry import register         # Auto-registration
```

---

## 6. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Forgot `sys.modules` registration | `from pydrology._core.<model> import ...` raises `ModuleNotFoundError` | Add `"<model>"` to the loop in `crates/pydrology-python/src/lib.rs` |
| `np.asarray()` instead of `np.ascontiguousarray()` | Rust panics on non-contiguous numpy arrays | Always use `np.ascontiguousarray(arr, dtype=np.float64)` in Python shim |
| Missing auto-register at bottom of shim | Model absent from `registry.list_models()` | Add `register("<model>", _self)` at end of `python/pydrology/models/<model>.py` |
| Missing import in `__init__.py` | Auto-register never triggers on `import pydrology` | Add `import pydrology.models.<model>` to `python/pydrology/__init__.py` |
| `FluxesTimeseriesOps` not impl'd | Compilation error: trait bound not satisfied | Impl all 6 methods: `with_capacity`, `push`, `len`, `is_empty`, `with_len`, `write_unchecked` |
| `pub mod <model>;` missing from `lib.rs` | `unresolved import` in Rust | Add to `crates/pydrology-core/src/lib.rs` alphabetically |
| State `Copy` on variable-size state | Compilation error: `Copy` requires fixed size | Only derive `Copy` if `FIXED_STATE`; otherwise use `Clone` only |
| `_model_fixtures` missing elif | `ValueError: Unknown model` in conformance tests | Add branch for new model in `tests/test_model_conformance.py` |
| Forgot `_make_initial_state` branch | Conformance `test_state_roundtrip` fails | Add branch in `_make_initial_state()` with correct init signature |
| Forgot `_reconstruct_state` branch | Conformance `test_state_roundtrip` fails | Add branch in `_reconstruct_state()` with correct `from_array` signature |
| `PARAM_BOUNDS` order mismatch | Calibration produces nonsense results | Ensure `PARAM_BOUNDS` slice order matches `PARAM_NAMES` order exactly |
| Fluxes struct missing `streamflow` field | Conformance tests fail on `result.fluxes.streamflow` | Every model's `Fluxes` must include a `streamflow: f64` field |

### Required Exports in Python Shim `__all__`

```python
__all__ = [
    "DEFAULT_BOUNDS",
    "<Model>Fluxes",      # e.g., GR2MFluxes
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "SUPPORTED_RESOLUTIONS",
    "run",
    "step",
]
```

### Conformance Test Contract

Every model registered in `ALL_MODELS` must satisfy:

- `model.PARAM_NAMES` -- tuple of parameter name strings
- `model.DEFAULT_BOUNDS` -- dict mapping each param name to `(min, max)`
- `model.STATE_SIZE` -- positive int
- `model.Parameters.from_array(arr)` -- construct from 1D numpy array
- `np.asarray(params)` -- convert back to 1D numpy array
- `model.State.initialize(params, ...)` -- create default initial state
- `np.asarray(state)` -- convert state to 1D numpy array
- `model.State.from_array(arr, ...)` -- reconstruct from 1D numpy array
- `model.run(params, forcing, ...)` -- returns `ModelOutput` with `.fluxes.streamflow`
- `model.run(...).fluxes.to_dict()` -- returns `dict[str, np.ndarray]`
