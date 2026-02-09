"""GR2M model orchestration functions.

This module provides the main entry points for running the GR2M model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR2MFluxes
from .types import Parameters, State

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData


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
    from pydrology._core.gr2m import gr2m_step

    new_state_arr, fluxes = gr2m_step(
        np.asarray(state), np.asarray(params), precip, pet
    )

    new_state = State.from_array(new_state_arr)
    return new_state, fluxes


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

    from pydrology._core.gr2m import gr2m_run
    from pydrology.outputs import ModelOutput

    state_arr = np.asarray(initial_state) if initial_state is not None else None

    result = gr2m_run(
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
