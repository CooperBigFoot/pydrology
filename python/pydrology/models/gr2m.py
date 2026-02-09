"""GR2M monthly rainfall-runoff model.

Public API for the GR2M hydrological model. Combines constants, types,
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

MAX_TANH_ARG: float = 13.0
ROUTING_DENOMINATOR: float = 60.0

PARAM_NAMES: tuple[str, ...] = ("x1", "x2")
DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x1": (1.0, 2500.0),
    "x2": (0.2, 2.0),
}
STATE_SIZE: int = 2
SUPPORTED_RESOLUTIONS: tuple[Resolution, ...] = (Resolution.monthly,)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parameters:
    """GR2M calibrated parameters.

    Both parameters that define the model behavior. This is a frozen dataclass
    to prevent accidental modification during simulation.

    Attributes:
        x1: Production store capacity [mm].
        x2: Groundwater exchange coefficient [-].
    """

    x1: float  # Production store capacity [mm]
    x2: float  # Groundwater exchange coefficient [-]

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert parameters to a 1D array for array protocol.

        Layout: [x1, x2] (2 elements)
        """
        arr = np.array([self.x1, self.x2], dtype=np.float64)
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
        )


@dataclass
class State:
    """GR2M model state variables.

    Mutable state that evolves during simulation. Contains the two stores.

    Attributes:
        production_store: S - soil moisture store level [mm].
        routing_store: R - groundwater/routing store level [mm].
    """

    production_store: float  # S - soil moisture [mm]
    routing_store: float  # R - groundwater [mm]

    @classmethod
    def initialize(cls, params: Parameters) -> State:
        """Create initial state from parameters.

        Uses standard initialization fractions:
        - Production store at 30% of X1 capacity
        - Routing store at 30% of X1 (since X2 is dimensionless)

        Args:
            params: Model parameters to derive initial state from.

        Returns:
            Initialized State object ready for simulation.
        """
        return cls(
            production_store=0.3 * params.x1,
            routing_store=0.3 * params.x1,
        )

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        """Convert state to a 1D array for array protocol.

        Layout: [production_store, routing_store]
        Total: 2 elements
        """
        arr = np.array([self.production_store, self.routing_store], dtype=np.float64)
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    @classmethod
    def from_array(cls, arr: np.ndarray) -> State:
        """Reconstruct State from array."""
        return cls(
            production_store=float(arr[0]),
            routing_store=float(arr[1]),
        )


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GR2MFluxes:
    """GR2M model flux outputs as arrays.

    All arrays have the same length as the input forcing data.
    Field order matches the airGR MISC array indices (1-11).

    Attributes:
        pet: Potential evapotranspiration [mm/month]. MISC(1)
        precip: Precipitation input [mm/month]. MISC(2)
        production_store: Production store level after timestep [mm]. MISC(3)
        rainfall_excess: Rainfall excess / net precipitation P1 [mm/month]. MISC(4)
        storage_fill: Storage infiltration PS [mm/month]. MISC(5)
        actual_et: Actual evapotranspiration AE [mm/month]. MISC(6)
        percolation: Percolation from production store P2 [mm/month]. MISC(7)
        routing_input: Total water to routing P3 = P1 + P2 [mm/month]. MISC(8)
        routing_store: Routing store level after timestep [mm]. MISC(9)
        exchange: Groundwater exchange AEXCH [mm/month]. MISC(10)
        streamflow: Total simulated streamflow Q [mm/month]. MISC(11)
    """

    pet: np.ndarray
    precip: np.ndarray
    production_store: np.ndarray
    rainfall_excess: np.ndarray
    storage_fill: np.ndarray
    actual_et: np.ndarray
    percolation: np.ndarray
    routing_input: np.ndarray
    routing_store: np.ndarray
    exchange: np.ndarray
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}


# ---------------------------------------------------------------------------
# Run / Step
# ---------------------------------------------------------------------------


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR2M model.

    Implements the complete GR2M algorithm:
    1. Production store update (rainfall neutralization)
    2. Evapotranspiration extraction
    3. Percolation from production store
    4. Route water through routing store with exchange
    5. Compute streamflow

    Args:
        state: Current model state (stores).
        params: Model parameters (X1, X2).
        precip: Monthly precipitation (mm/month).
        pet: Monthly potential evapotranspiration (mm/month).

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs
    """
    from pydrology._core import gr2m as _rust

    new_state_arr, fluxes = _rust.gr2m_step(np.asarray(state), np.asarray(params), precip, pet)

    new_state = State.from_array(new_state_arr)
    fluxes_converted: dict[str, float] = {k: float(v) for k, v in fluxes.items()}
    return new_state, fluxes_converted


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
) -> ModelOutput[GR2MFluxes]:
    """Run the GR2M model over a timeseries.

    Executes the GR2M model for each timestep in the input forcing data, returning
    a ModelOutput with all model outputs.

    Args:
        params: Model parameters (X1, X2).
        forcing: Input forcing data with precip and pet arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).

    Returns:
        ModelOutput containing GR2M flux outputs.
        Access streamflow via result.streamflow or result.fluxes.streamflow (numpy array).
        Convert to DataFrame via result.to_dataframe().

    Raises:
        ValueError: If forcing resolution is not monthly.
    """
    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR2M supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    from pydrology._core import gr2m as _rust
    from pydrology.outputs import ModelOutput

    state_arr = np.asarray(initial_state) if initial_state is not None else None

    result = _rust.gr2m_run(
        np.asarray(params),
        forcing.precip.astype(np.float64),
        forcing.pet.astype(np.float64),
        state_arr,
    )

    gr2m_fluxes = GR2MFluxes(**result)

    return ModelOutput(
        time=forcing.time,
        fluxes=gr2m_fluxes,
        snow=None,
        snow_layers=None,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_BOUNDS",
    "GR2MFluxes",
    "PARAM_NAMES",
    "Parameters",
    "STATE_SIZE",
    "State",
    "SUPPORTED_RESOLUTIONS",
    "run",
    "step",
]

# Auto-register
import pydrology.models.gr2m as _self  # noqa: E402
from pydrology.registry import register  # noqa: E402

register("gr2m", _self)
