# Pydrology Design Document

> Design decisions for transforming GR6J into `pydrology` — a unified Python library for lumped hydrological models.

---

## Vision

A Python library that implements multiple lumped hydrological models with a consistent interface, enabling easy benchmarking against deep learning models.

```python
import pydrology as phy

results = phy.benchmark(
    models=["gr6j", "hbv", "gr6j_cemaneige"],
    forcing=forcing,
    observed=observed,
    objectives=["nse", "kge"],
)
```

---

## Core Design Principles

| Principle | Decision |
|-----------|----------|
| **Functional over OOP** | No class hierarchies. Dataclasses for data, functions for logic. |
| **Each combination is a model** | No plugins, no composition. `gr6j_cemaneige` is a standalone model. |
| **Performance via Numba** | Dual-layer API: Pythonic wrappers over `@njit` kernels. |
| **Type safety** | Frozen dataclasses with `__array__()` / `from_array()` for IDE support. |
| **Model-agnostic calibration** | Uses `PARAM_NAMES` and module introspection, not per-model code. |

---

## Repository Structure

```
src/pydrology/
├── __init__.py                    # Top-level API, registry access
├── registry.py                    # Model registry
├── types.py                       # Shared types (ForcingData, Catchment, HydroOutput protocol)
├── outputs.py                     # ModelOutput generic, base flux protocol
│
├── processes/                     # Shared Numba process functions
│   ├── gr6j.py                    # GR6J equations (production_store_update, etc.)
│   ├── cemaneige.py               # CemaNeige equations (snow_melt, etc.)
│   ├── hbv.py                     # HBV equations
│   └── ...
│
├── models/
│   ├── gr4j/
│   │   ├── __init__.py            # Exports: Parameters, State, run, step, PARAM_NAMES, etc.
│   │   ├── types.py               # Parameters, State dataclasses
│   │   └── run.py                 # run(), step(), _run_numba(), _step_numba()
│   │
│   ├── gr6j/
│   │   └── ...
│   │
│   ├── gr6j_cemaneige/            # GR6J + CemaNeige as ONE model
│   │   ├── __init__.py
│   │   ├── types.py               # Parameters includes ctg, kf
│   │   └── run.py                 # Wires CemaNeige → GR6J
│   │
│   ├── hbv/                       # Snow built-in (tt, cfmax, etc. in Parameters)
│   │   └── ...
│   │
│   └── sac_sma/
│       └── ...
│
├── calibration/
│   ├── __init__.py
│   ├── calibrate.py               # Model-agnostic calibration
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── registry.py            # @register decorator
│   │   └── functions.py           # nse, kge, log_nse, etc.
│   └── types.py                   # ObservedData, Solution
│
└── benchmark/
    ├── __init__.py
    └── runner.py                  # benchmark() function
```

---

## The Model Contract

Every model module **must** export these symbols:

### Module-Level Constants

```python
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (0.01, 20.0),
}

STATE_SIZE: int = 63  # Total floats in flattened state array
```

### Parameters (Frozen Dataclass)

```python
@dataclass(frozen=True)
class Parameters:
    """Model parameters. Immutable."""
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert to 1D array in PARAM_NAMES order."""
        arr = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6])
        return arr if dtype is None else arr.astype(dtype)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct from array."""
        return cls(
            x1=float(arr[0]),
            x2=float(arr[1]),
            x3=float(arr[2]),
            x4=float(arr[3]),
            x5=float(arr[4]),
            x6=float(arr[5]),
        )
```

### State (Mutable Dataclass)

```python
@dataclass
class State:
    """Model state. Evolves during simulation."""
    production_store: float
    routing_store: float
    exponential_store: float
    uh1_states: np.ndarray
    uh2_states: np.ndarray

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Create default initial state derived from parameters."""
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(20),
            uh2_states=np.zeros(40),
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Flatten to 1D array (STATE_SIZE elements)."""
        ...

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Reconstruct from flattened array."""
        ...
```

### Functions

```python
def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    # Model-specific precomputed values (e.g., UH ordinates)
    **precomputed,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep. Returns (new_state, fluxes_dict)."""
    ...


def run(
    params: Parameters,
    forcing: ForcingData,
    catchment: Catchment | None = None,
    initial_state: State | None = None,
) -> ModelOutput[Fluxes]:
    """Run model over full timeseries."""
    ...


@njit(cache=True)
def _step_numba(
    state_arr: np.ndarray,    # Modified in place
    params_arr: np.ndarray,
    precip: float,
    pet: float,
    output_arr: np.ndarray,   # Written here
) -> None:
    """Numba kernel for single timestep."""
    ...


@njit(cache=True)
def _run_numba(
    state_arr: np.ndarray,
    params_arr: np.ndarray,
    precip_arr: np.ndarray,
    pet_arr: np.ndarray,
    outputs_arr: np.ndarray,
) -> None:
    """Numba kernel for full timeseries."""
    ...
```

---

## Output Design

### The Protocol

```python
class HydroOutput(Protocol):
    """Contract for all model flux outputs."""

    @property
    def streamflow(self) -> np.ndarray:
        """Simulated streamflow [mm/day]. Required."""
        ...

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert all fluxes to dictionary."""
        ...
```

### Model-Specific Fluxes

```python
@dataclass(frozen=True)
class GR6JFluxes:
    """GR6J-specific flux outputs. Implements HydroOutput protocol."""
    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    routing_store: np.ndarray
    exponential_store: np.ndarray
    # ... all 20 fields
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class HBVFluxes:
    """HBV-specific flux outputs."""
    soil_moisture: np.ndarray
    upper_zone: np.ndarray
    lower_zone: np.ndarray
    # ... HBV fields
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
```

### Generic Output Container

```python
FluxT = TypeVar("FluxT", bound=HydroOutput)


@dataclass(frozen=True)
class ModelOutput(Generic[FluxT]):
    """Complete model output with time index."""
    time: np.ndarray
    fluxes: FluxT

    @property
    def streamflow(self) -> np.ndarray:
        """Convenience accessor."""
        return self.fluxes.streamflow

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with time index."""
        df = pd.DataFrame(self.fluxes.to_dict(), index=self.time)
        df.index.name = "time"
        return df


# Type aliases for convenience
GR6JOutput = ModelOutput[GR6JFluxes]
HBVOutput = ModelOutput[HBVFluxes]
```

### Usage

```python
# Model-agnostic (calibration code)
output = model.run(params, forcing)
sim = output.streamflow  # Works for ANY model

# Model-specific (analysis code) — IDE autocomplete works
output: GR6JOutput = gr6j.run(params, forcing)
output.fluxes.exponential_store  # GR6J-specific field
```

---

## No Composition — Variants Instead

### The Problem with Composition

```python
# BAD: Composition doesn't scale
params = gr6j.Parameters(
    x1=350, ...,
    snow=CemaNeige(...),          # What about multiple?
    groundwater=SomePlugin(...),  # This gets messy
    evap=AnotherPlugin(...),
)
```

### The Solution: Each Combination is a Model

```python
# GOOD: Explicit model variants
from pydrology.models import gr6j, gr6j_cemaneige

# Base GR6J (no snow)
output = gr6j.run(params, forcing)

# GR6J + CemaNeige (different Parameters class)
output = gr6j_cemaneige.run(params, forcing, catchment=catchment)
```

### Code Sharing via Process Functions

Models share underlying Numba functions:

```python
# pydrology/processes/gr6j.py
@njit(cache=True)
def production_store_update(precip, pet, store, x1): ...

@njit(cache=True)
def routing_store_update(store, inflow, exchange, x3): ...


# pydrology/processes/cemaneige.py
@njit(cache=True)
def snow_accumulation(precip, temp, pack): ...

@njit(cache=True)
def snow_melt(pack, temp, kf): ...
```

```python
# gr6j_cemaneige/run.py — wires shared functions together
from pydrology.processes.gr6j import production_store_update, routing_store_update
from pydrology.processes.cemaneige import snow_accumulation, snow_melt

@njit(cache=True)
def _run_numba(...):
    # CemaNeige first
    precip = snow_melt(snow_accumulation(...), ...)
    # Then GR6J
    production_store_update(precip, ...)
    routing_store_update(...)
```

---

## Calibration

### Signature

```python
def calibrate(
    model: str,                                    # "gr6j", "gr6j_cemaneige", etc.
    forcing: ForcingData,
    observed: ObservedData,
    objectives: list[str],
    bounds: dict[str, tuple[float, float]] | None = None,
    use_default_bounds: bool = False,
    catchment: Catchment | None = None,
    initial_state: State | None = None,
    warmup: int = 365,
    population_size: int = 50,
    generations: int = 100,
    seed: int | None = None,
    n_workers: int = 1,
) -> Solution | list[Solution]:
    """Model-agnostic calibration using evolutionary algorithms."""
    ...
```

### Bounds Logic

```python
# Inside calibrate()
model_module = get_model(model)  # From registry

if bounds is not None:
    # User provided bounds — validate coverage
    missing = set(model_module.PARAM_NAMES) - set(bounds.keys())
    if missing:
        raise ValueError(f"Missing bounds for: {missing}")
    final_bounds = bounds

elif use_default_bounds:
    # Use model defaults
    final_bounds = model_module.DEFAULT_BOUNDS

else:
    raise ValueError("Must provide bounds or set use_default_bounds=True")
```

### No Validation at Parameter Construction

Parameters can be created with any values:

```python
# This is fine — no validation
params = gr6j.Parameters(x1=9999, x2=-100, ...)
output = gr6j.run(params, forcing)  # Model runs (might produce nonsense)

# Validation happens at calibration time
result = calibrate(
    model="gr6j",
    bounds={"x1": (1, 2500), ...},  # Optimizer stays within these
    ...
)
```

---

## Registry

```python
# pydrology/registry.py
from types import ModuleType

_models: dict[str, ModuleType] = {}


def register(name: str, module: ModuleType) -> None:
    """Register a model module. Validates contract."""
    required = ("PARAM_NAMES", "DEFAULT_BOUNDS", "STATE_SIZE",
                "Parameters", "State", "run", "step")

    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise ValueError(f"Model '{name}' missing exports: {missing}")

    # Validate Parameters has required methods
    if not hasattr(module.Parameters, "from_array"):
        raise ValueError(f"Model '{name}' Parameters missing from_array()")
    if not hasattr(module.Parameters, "__array__"):
        raise ValueError(f"Model '{name}' Parameters missing __array__()")

    _models[name] = module


def get_model(name: str) -> ModuleType:
    """Get a registered model by name."""
    if name not in _models:
        raise KeyError(f"Unknown model: '{name}'. Available: {list(_models)}")
    return _models[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(_models.keys())
```

### Auto-Registration

```python
# pydrology/models/gr6j/__init__.py
from .types import Parameters, State
from .run import run, step, _run_numba, _step_numba

PARAM_NAMES = ("x1", "x2", "x3", "x4", "x5", "x6")
DEFAULT_BOUNDS = {"x1": (1, 2500), ...}
STATE_SIZE = 63

# Self-register on import
from pydrology.registry import register
import pydrology.models.gr6j as _self
register("gr6j", _self)
```

---

## Catchment

Kept as a shared type for models that need physical catchment properties:

```python
# pydrology/types.py
@dataclass(frozen=True)
class Catchment:
    """Static physical properties of a catchment.

    Used by models with snow components or elevation-dependent processes.
    """
    mean_annual_solid_precip: float          # For snow initialization [mm/year]
    hypsometric_curve: np.ndarray | None = None  # Elevation distribution [m]
    input_elevation: float | None = None     # Met station elevation [m]
    n_layers: int = 1                        # Elevation bands for snow
    temp_gradient: float = 0.6               # Temperature lapse rate [°C/100m]
    precip_gradient: float = 0.0004          # Precipitation gradient [1/m]
```

Models that need it validate internally:

```python
# gr6j_cemaneige/run.py
def run(params, forcing, catchment=None, initial_state=None):
    if catchment is None:
        raise ValueError("gr6j_cemaneige requires catchment for snow initialization")
    ...
```

---

## Planned Models

| Model | Parameters | Notes |
|-------|------------|-------|
| `gr4j` | 4 | Base GR model |
| `gr6j` | 6 | GR4J + exponential store |
| `gr6j_cemaneige` | 8 | GR6J + CemaNeige snow |
| `hbv` | ~15 | Snow built-in |
| `sac_sma` | ~16 | Sacramento model |
| `sac_sma_snow17` | ~20+ | SAC-SMA + SNOW-17 |
| `simhyd` | ~9 | Australian model |
| `hymod` | 5 | Simple, popular in UQ |

---

## Summary of Key Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Architecture | Functional (dataclasses + functions) | Matches existing GR6J style |
| Plugins/Composition | **None** — each combo is a model | Avoids complexity, scales cleanly |
| Parameters | Frozen dataclass with `from_array()` | Type safety + calibration support |
| State | Mutable dataclass with `initialize(params)` | Needs to evolve; single factory |
| Output | Protocol + Generic `ModelOutput[FluxT]` | Model-agnostic AND model-specific |
| Bounds validation | At calibration, not construction | Allow any params for sensitivity analysis |
| `use_default_bounds` | Flag, defaults to `False` | Explicit over implicit |
| `run()` signature | Identical across models | `catchment` optional, validated internally |
| Code sharing | Shared process functions | No duplication of math |
| Registry | Auto-registration on import | Seamless discovery |

---

## Next Steps

1. **Rename repo** GR6J → pydrology (GitHub Settings)
2. **Restructure** `src/gr6j/` → `src/pydrology/models/gr6j/`
3. **Extract** process functions to `src/pydrology/processes/`
4. **Add** registry and auto-registration
5. **Create** `gr6j_cemaneige` as separate model
6. **Update** calibration to be model-agnostic
7. **Implement** HBV as proof of generalization
