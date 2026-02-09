"""GR6J daily rainfall-runoff model.

Public API for the GR6J hydrological model. Combines constants, types,
outputs, and run/step functions into a single module.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from pydrology.types import Resolution

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Routing split fractions
B: float = 0.9  # Fraction of PR to UH1 (slow branch)
C: float = 0.4  # Fraction of UH1 output to exponential store

# Unit hydrograph parameters
D: float = 2.5  # S-curve exponent
NH: int = 20  # UH1 length (days), UH2 is 2*NH = 40 days

# Percolation constant: (9/4)^4 = 2.25^4
PERC_CONSTANT: float = 25.62890625

# Numerical safeguards to prevent overflow
MAX_TANH_ARG: float = 13.0
MAX_EXP_ARG: float = 33.0
EXP_BRANCH_THRESHOLD: float = 7.0

# Model contract constants
PARAM_NAMES: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6")
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (-5.0, 5.0),
    "x3": (1.0, 1000.0),
    "x4": (0.5, 10.0),
    "x5": (-4.0, 4.0),
    "x6": (1.0, 50.0),
}
STATE_SIZE: int = 63
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.daily,)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameters:
    """GR6J calibrated parameters.

    All 6 parameters that define the model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Attributes:
        x1: Production store capacity [mm].
        x2: Intercatchment exchange coefficient [mm/day].
        x3: Routing store capacity [mm].
        x4: Unit hydrograph time constant [days].
        x5: Intercatchment exchange threshold [-].
        x6: Exponential store scale parameter [mm].
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Intercatchment exchange coefficient [mm/day]
    x3: float  # Routing store capacity [mm]
    x4: float  # Unit hydrograph time constant [days]
    x5: float  # Intercatchment exchange threshold [-]
    x6: float  # Exponential store scale parameter [mm]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for array protocol.

        Layout: [x1, x2, x3, x4, x5, x6] (6 elements)
        """
        arr = np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> Parameters:
        """Reconstruct Parameters from array."""
        if len(arr) != len(PARAM_NAMES):
            msg = f"Expected array of length {len(PARAM_NAMES)}, got {len(arr)}"
            raise ValueError(msg)
        return cls(
            x1=float(arr[0]),
            x2=float(arr[1]),
            x3=float(arr[2]),
            x4=float(arr[3]),
            x5=float(arr[4]),
            x6=float(arr[5]),
        )


@dataclass
class State:
    """GR6J model state variables.

    Mutable state that evolves during simulation. Contains the three stores
    and the unit hydrograph convolution states.

    Attributes:
        production_store: S - soil moisture store level [mm].
        routing_store: R - groundwater/routing store level [mm].
        exponential_store: Exp - slow drainage store, can be negative [mm].
        uh1_states: Convolution states for UH1 (20 elements).
        uh2_states: Convolution states for UH2 (40 elements).
    """

    production_store: float  # S - soil moisture [mm]
    routing_store: float  # R - groundwater [mm]
    exponential_store: float  # Exp - slow drainage, can be negative [mm]
    uh1_states: np.ndarray  # 20-element array for UH1
    uh2_states: np.ndarray  # 40-element array for UH2

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Create initial state from parameters.

        Uses standard initialization fractions:
        - Production store at 30% capacity
        - Routing store at 50% capacity
        - Exponential store at zero
        - Unit hydrograph states all zero

        Args:
            params: Model parameters to derive initial state from.

        Returns:
            Initialized State object ready for simulation.
        """
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.5 * params.x3,
            exponential_store=0.0,
            uh1_states=np.zeros(NH),
            uh2_states=np.zeros(2 * NH),
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

        Layout: [production_store, routing_store, exponential_store, uh1_states[0:20], uh2_states[0:40]]
        Total: 63 elements
        """
        arr = np.empty(63, dtype=np.float64)
        arr[0] = self.production_store
        arr[1] = self.routing_store
        arr[2] = self.exponential_store
        arr[3:23] = self.uh1_states
        arr[23:63] = self.uh2_states
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Reconstruct State from array."""
        return cls(
            production_store=float(arr[0]),
            routing_store=float(arr[1]),
            exponential_store=float(arr[2]),
            uh1_states=arr[3:23].copy(),
            uh2_states=arr[23:63].copy(),
        )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GR6JFluxes:
    """GR6J model flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
        pet: Potential evapotranspiration [mm/day].
        precip: Precipitation input to GR6J [mm/day]. When snow module is
            enabled, this is the liquid water output from CemaNeige.
        production_store: Production store level after timestep [mm].
        net_rainfall: Net rainfall after interception [mm/day].
        storage_infiltration: Water infiltrating to production store [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        percolation: Percolation from production store [mm/day].
        effective_rainfall: Total effective rainfall after percolation [mm/day].
        q9: Output from UH1 (90% branch) [mm/day].
        q1: Output from UH2 (10% branch) [mm/day].
        routing_store: Routing store level after timestep [mm].
        exchange: Groundwater exchange potential [mm/day].
        actual_exchange_routing: Actual exchange from routing store [mm/day].
        actual_exchange_direct: Actual exchange from direct branch [mm/day].
        actual_exchange_total: Total actual exchange [mm/day].
        qr: Outflow from routing store [mm/day].
        qrexp: Outflow from exponential store [mm/day].
        exponential_store: Exponential store level after timestep [mm].
        qd: Direct branch outflow [mm/day].
        streamflow: Total simulated streamflow [mm/day].
    """

    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    net_rainfall: np.ndarray
    storage_infiltration: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    effective_rainfall: np.ndarray
    q9: np.ndarray
    q1: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    actual_exchange_routing: np.ndarray
    actual_exchange_direct: np.ndarray
    actual_exchange_total: np.ndarray
    qr: np.ndarray
    qrexp: np.ndarray
    exponential_store: np.ndarray
    qd: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


# Backward compatibility alias
GR6JOutput = GR6JFluxes


# ---------------------------------------------------------------------------
# Run / Step
# ---------------------------------------------------------------------------


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J model.

    Args:
        state: Current model state (stores and UH states).
        params: Model parameters (X1-X6).
        precip: Daily precipitation (mm/day).
        pet: Daily potential evapotranspiration (mm/day).
        uh1_ordinates: Pre-computed UH1 ordinates from compute_uh_ordinates().
        uh2_ordinates: Pre-computed UH2 ordinates from compute_uh_ordinates().

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs
    """
    from pydrology._core import gr6j as _rust

    state_arr = np.asarray(state, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)

    new_state_arr, fluxes_dict = _rust.gr6j_step(
        state_arr,
        params_arr,
        precip,
        pet,
        np.ascontiguousarray(uh1_ordinates, dtype=np.float64),
        np.ascontiguousarray(uh2_ordinates, dtype=np.float64),
    )

    new_state = State.from_array(new_state_arr)

    # Convert numpy scalars to Python floats for consistency
    fluxes: dict[str, float] = {k: float(v) for k, v in fluxes_dict.items()}

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
) -> ModelOutput[GR6JFluxes]:
    """Run the GR6J model over a timeseries.

    Args:
        params: Model parameters (X1-X6).
        forcing: Input forcing data with precip and pet arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).

    Returns:
        ModelOutput containing GR6J flux outputs.
    """
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    from pydrology._core import gr6j as _rust
    from pydrology.outputs import ModelOutput

    params_arr = np.asarray(params, dtype=np.float64)

    initial_state_arr = None
    if initial_state is not None:
        initial_state_arr = np.asarray(initial_state, dtype=np.float64)

    result = _rust.gr6j_run(
        params_arr,
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        initial_state_arr,
    )

    gr6j_fluxes = GR6JFluxes(
        pet=result["pet"],
        precip=result["precip"],
        production_store=result["production_store"],
        net_rainfall=result["net_rainfall"],
        storage_infiltration=result["storage_infiltration"],
        actual_et=result["actual_et"],
        percolation=result["percolation"],
        effective_rainfall=result["effective_rainfall"],
        q9=result["q9"],
        q1=result["q1"],
        routing_store=result["routing_store"],
        exchange=result["exchange"],
        actual_exchange_routing=result["actual_exchange_routing"],
        actual_exchange_direct=result["actual_exchange_direct"],
        actual_exchange_total=result["actual_exchange_total"],
        qr=result["qr"],
        qrexp=result["qrexp"],
        exponential_store=result["exponential_store"],
        qd=result["qd"],
        streamflow=result["streamflow"],
    )

    return ModelOutput(
        time=forcing.time,
        fluxes=gr6j_fluxes,
        snow=None,
        snow_layers=None,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_BOUNDS",
    "GR6JFluxes",
    "GR6JOutput",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "SUPPORTED_RESOLUTIONS",
    "run",
    "step",
]

# Auto-register
import pydrology.models.gr6j as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("gr6j", _self)
