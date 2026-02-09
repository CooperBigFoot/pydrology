"""GR6J model orchestration functions.

This module provides the main entry points for running the GR6J model:
- step(): Execute a single timestep (Rust backend)
- run(): Execute the model over a timeseries (Rust backend)

Also provides _step_numba for backward compatibility with GR6J-CemaNeige
coupled model until Phase 4 ports CemaNeige to Rust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# ruff: noqa: I001
# Import order matters: _compat must patch numpy before numba import
import pydrology._compat  # noqa: F401
import numpy as np
from numba import njit

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR6JFluxes
from .processes import (
    direct_branch,
    exponential_store_update,
    groundwater_exchange,
    percolation,
    production_store_update,
    routing_store_update,
)
from .types import Parameters, State

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import ForcingData

# Constants inlined for Numba compatibility
_B: float = 0.9  # Fraction of PR to UH1 (slow branch)
_C: float = 0.4  # Fraction of UH1 output to exponential store


@njit(cache=True)
def _step_numba(
    state_arr: np.ndarray,  # shape (63,) - modified in place
    params_arr: np.ndarray,  # shape (6,)
    precip: float,
    pet: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    output_arr: np.ndarray,  # shape (20,) - output written here
) -> None:
    """Execute one timestep of GR6J using arrays (Numba-optimized).

    Retained for backward compatibility with GR6J-CemaNeige coupled model.

    State layout: [production_store, routing_store, exponential_store, uh1_states[20], uh2_states[40]]
    Params layout: [x1, x2, x3, x4, x5, x6]
    Output layout: [pet, precip, production_store, net_rainfall, storage_infiltration,
                    actual_et, percolation, effective_rainfall, q9, q1, routing_store,
                    exchange, actual_exchange_routing, actual_exchange_direct,
                    actual_exchange_total, qr, qrexp, exponential_store, qd, streamflow]
    """
    # Unpack parameters
    x1 = params_arr[0]
    x2 = params_arr[1]
    x3 = params_arr[2]
    # x4 is not used in step (only for UH computation, which is done once)
    x5 = params_arr[4]
    x6 = params_arr[5]

    # Unpack state
    production_store = state_arr[0]
    routing_store = state_arr[1]
    exponential_store = state_arr[2]

    # 1. Production store update
    prod_store_after_ps, actual_et, net_rainfall_pn, effective_rainfall_pr = production_store_update(
        precip, pet, production_store, x1
    )

    # Compute storage infiltration for output
    storage_infiltration = net_rainfall_pn - effective_rainfall_pr if precip >= pet else 0.0

    # 2. Percolation
    prod_store_after_perc, percolation_amount = percolation(prod_store_after_ps, x1)

    # Add percolation to effective rainfall
    total_effective_rainfall = effective_rainfall_pr + percolation_amount

    # 3. Split to unit hydrographs
    uh1_input = _B * total_effective_rainfall
    uh2_input = (1.0 - _B) * total_effective_rainfall

    # 4. Convolve through unit hydrographs
    q9 = state_arr[3]
    new_uh1 = np.zeros(20)
    for k in range(19):
        new_uh1[k] = state_arr[4 + k] + uh1_ordinates[k] * uh1_input
    new_uh1[19] = uh1_ordinates[19] * uh1_input

    q1 = state_arr[23]
    new_uh2 = np.zeros(40)
    for k in range(39):
        new_uh2[k] = state_arr[24 + k] + uh2_ordinates[k] * uh2_input
    new_uh2[39] = uh2_ordinates[39] * uh2_input

    # 5. Groundwater exchange
    exchange_f = groundwater_exchange(routing_store, x2, x3, x5)

    # 6. Update routing store
    routing_input = (1.0 - _C) * q9
    new_routing_store, qr, actual_exchange_routing = routing_store_update(routing_store, routing_input, exchange_f, x3)

    # 7. Update exponential store
    exp_input = _C * q9
    new_exp_store, qrexp = exponential_store_update(exponential_store, exp_input, exchange_f, x6)

    # 8. Direct branch
    qd, actual_exchange_direct = direct_branch(q1, exchange_f)

    # 9. Total streamflow
    streamflow = max(qr + qrexp + qd, 0.0)

    # Total actual exchange
    actual_exchange_total = actual_exchange_routing + actual_exchange_direct + exchange_f

    # Update state array in place
    state_arr[0] = prod_store_after_perc
    state_arr[1] = new_routing_store
    state_arr[2] = new_exp_store
    for k in range(20):
        state_arr[3 + k] = new_uh1[k]
    for k in range(40):
        state_arr[23 + k] = new_uh2[k]

    # Write outputs
    output_arr[0] = pet
    output_arr[1] = precip
    output_arr[2] = prod_store_after_perc
    output_arr[3] = net_rainfall_pn
    output_arr[4] = storage_infiltration
    output_arr[5] = actual_et
    output_arr[6] = percolation_amount
    output_arr[7] = total_effective_rainfall
    output_arr[8] = q9
    output_arr[9] = q1
    output_arr[10] = new_routing_store
    output_arr[11] = exchange_f
    output_arr[12] = actual_exchange_routing
    output_arr[13] = actual_exchange_direct
    output_arr[14] = actual_exchange_total
    output_arr[15] = qr
    output_arr[16] = qrexp
    output_arr[17] = new_exp_store
    output_arr[18] = qd
    output_arr[19] = streamflow


@njit(cache=True)
def _run_numba(
    state_arr: np.ndarray,  # shape (63,)
    params_arr: np.ndarray,  # shape (6,)
    precip_arr: np.ndarray,  # shape (n_timesteps,)
    pet_arr: np.ndarray,  # shape (n_timesteps,)
    uh1_ordinates: np.ndarray,  # shape (20,)
    uh2_ordinates: np.ndarray,  # shape (40,)
    outputs_arr: np.ndarray,  # shape (n_timesteps, 20)
) -> None:
    """Run GR6J over a timeseries using arrays (Numba-optimized).

    Retained for backward compatibility with GR6J-CemaNeige coupled model.
    State is modified in place. Outputs are written to outputs_arr.
    """
    n_timesteps = len(precip_arr)
    output_single = np.zeros(20)

    for t in range(n_timesteps):
        _step_numba(
            state_arr,
            params_arr,
            precip_arr[t],
            pet_arr[t],
            uh1_ordinates,
            uh2_ordinates,
            output_single,
        )
        for i in range(20):
            outputs_arr[t, i] = output_single[i]


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
    import pydrology._core

    state_arr = np.asarray(state, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)

    new_state_arr, fluxes_dict = pydrology._core.gr6j.gr6j_step(
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

    import pydrology._core
    from pydrology.outputs import ModelOutput

    params_arr = np.asarray(params, dtype=np.float64)

    initial_state_arr = None
    if initial_state is not None:
        initial_state_arr = np.asarray(initial_state, dtype=np.float64)

    result = pydrology._core.gr6j.gr6j_run(
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
