"""GR6J-CemaNeige coupled model orchestration functions (Rust backend).

This module provides the main entry points for running the coupled GR6J-CemaNeige model:
- step(): Execute a single timestep (with snow processing)
- run(): Execute the model over a timeseries

Design: Unified multi-layer approach. If catchment.input_elevation is None,
extrapolation is skipped (single-layer behavior). Single code path for all cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pydrology.utils.elevation import (
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
)
from pydrology.models.cemaneige.outputs import SnowLayerOutputs, SnowOutput

from .constants import SUPPORTED_RESOLUTIONS
from .outputs import GR6JCemaNeigeFluxes
from .types import Parameters, State

if TYPE_CHECKING:
    from pydrology.outputs import ModelOutput
    from pydrology.types import Catchment, ForcingData


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    uh1_ordinates: np.ndarray,
    uh2_ordinates: np.ndarray,
    layer_elevations: np.ndarray | None = None,
    layer_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the GR6J-CemaNeige coupled model.

    Processes snow accumulation/melt through CemaNeige, then routes
    the liquid water through GR6J.

    Args:
        state: Current combined model state.
        params: Model parameters (8 total).
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily air temperature [deg C].
        uh1_ordinates: Pre-computed UH1 ordinates.
        uh2_ordinates: Pre-computed UH2 ordinates.
        layer_elevations: Elevation of each layer [m]. Required for multi-layer.
        layer_fractions: Area fraction of each layer [-]. Required for multi-layer.
        input_elevation: Elevation of forcing data [m]. If None, skip extrapolation.
        temp_gradient: Temperature lapse rate [deg C/100m]. Default 0.6.
        precip_gradient: Precipitation gradient [m^-1]. Default 0.00041.

    Returns:
        Tuple of (new_state, fluxes) where fluxes contains all snow and GR6J outputs.
    """
    import pydrology._core

    n_layers = state.n_layers

    # Set defaults for layer config
    if layer_elevations is None:
        layer_elevations = np.zeros(n_layers)
    if layer_fractions is None:
        layer_fractions = np.ones(n_layers) / n_layers

    # Convert state and params to arrays
    state_arr = np.ascontiguousarray(np.asarray(state), dtype=np.float64)
    params_arr = np.ascontiguousarray(np.asarray(params), dtype=np.float64)

    new_state_arr, fluxes_dict = pydrology._core.cemaneige.gr6j_cemaneige_step(
        state_arr,
        params_arr,
        precip,
        pet,
        temp,
        np.ascontiguousarray(uh1_ordinates, dtype=np.float64),
        np.ascontiguousarray(uh2_ordinates, dtype=np.float64),
        np.ascontiguousarray(layer_elevations, dtype=np.float64),
        np.ascontiguousarray(layer_fractions, dtype=np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
    )

    # Reconstruct state
    new_state = State.from_array(new_state_arr, n_layers)

    # Convert numpy scalars to Python floats for consistency
    fluxes: dict[str, float] = {k: float(v) for k, v in fluxes_dict.items()}

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    catchment: Catchment,
    initial_state: State | None = None,
) -> ModelOutput[GR6JCemaNeigeFluxes]:
    """Run the GR6J-CemaNeige coupled model over a timeseries.

    Executes the coupled snow-hydrology model for each timestep in the input
    forcing data. Snow is processed through CemaNeige (with optional multi-layer
    elevation bands), and the liquid water output is routed through GR6J.

    Args:
        params: Model parameters (8 total: 6 GR6J + 2 CemaNeige).
        forcing: Input forcing data with precip, pet, and temp arrays.
        catchment: Catchment properties (mean_annual_solid_precip, elevation config).
        initial_state: Initial model state. If None, uses State.initialize().

    Returns:
        ModelOutput containing GR6JCemaNeigeFluxes, SnowOutput, and optionally
        SnowLayerOutputs (for multi-layer simulations).

    Raises:
        ValueError: If forcing.temp is None (temperature required for snow module).
    """
    import pydrology._core
    from pydrology.outputs import ModelOutput

    # Validate temperature is provided
    if forcing.temp is None:
        msg = "Temperature data (forcing.temp) is required for the GR6J-CemaNeige model"
        raise ValueError(msg)

    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"GR6J-CemaNeige supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    n_timesteps = len(forcing)
    n_layers = catchment.n_layers

    # Derive layer properties if multi-layer
    if n_layers > 1 and catchment.hypsometric_curve is not None:
        layer_elevations, layer_fractions = derive_layers(catchment.hypsometric_curve, n_layers)
    else:
        # Single layer: use dummy values (extrapolation will be skipped)
        layer_elevations = np.zeros(n_layers)
        layer_fractions = np.ones(n_layers) / n_layers

    # Initialize state if not provided
    state = State.initialize(params, catchment) if initial_state is None else initial_state

    # Set gradients (use defaults if not specified)
    temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT
    precip_gradient = catchment.precip_gradient if catchment.precip_gradient is not None else GRAD_P_DEFAULT

    # Use NaN to signal no extrapolation
    input_elevation = catchment.input_elevation if catchment.input_elevation is not None else float("nan")

    # Convert to arrays
    initial_state_arr = np.ascontiguousarray(np.asarray(state), dtype=np.float64)
    params_arr = np.ascontiguousarray(np.asarray(params), dtype=np.float64)

    # Call Rust backend
    snow_result, gr6j_result, layer_result = pydrology._core.cemaneige.gr6j_cemaneige_run(
        params_arr,
        np.ascontiguousarray(forcing.precip, dtype=np.float64),
        np.ascontiguousarray(forcing.pet, dtype=np.float64),
        np.ascontiguousarray(forcing.temp, dtype=np.float64),
        initial_state_arr,
        None,  # uh1_ordinates (computed internally)
        None,  # uh2_ordinates (computed internally)
        n_layers,
        np.ascontiguousarray(layer_elevations, dtype=np.float64),
        np.ascontiguousarray(layer_fractions, dtype=np.float64),
        input_elevation,
        temp_gradient,
        precip_gradient,
        catchment.mean_annual_solid_precip,
    )

    # Build combined flux output
    combined_fluxes = GR6JCemaNeigeFluxes(
        # Snow outputs (10 fields)
        precip_raw=forcing.precip.copy(),
        snow_pliq=snow_result["snow_pliq"],
        snow_psol=snow_result["snow_psol"],
        snow_pack=snow_result["snow_pack"],
        snow_thermal_state=snow_result["snow_thermal_state"],
        snow_gratio=snow_result["snow_gratio"],
        snow_pot_melt=snow_result["snow_pot_melt"],
        snow_melt=snow_result["snow_melt"],
        snow_pliq_and_melt=snow_result["snow_pliq_and_melt"],
        snow_temp=snow_result["snow_temp"],
        # GR6J outputs (20 fields)
        pet=gr6j_result["pet"],
        precip=gr6j_result["precip"],
        production_store=gr6j_result["production_store"],
        net_rainfall=gr6j_result["net_rainfall"],
        storage_infiltration=gr6j_result["storage_infiltration"],
        actual_et=gr6j_result["actual_et"],
        percolation=gr6j_result["percolation"],
        effective_rainfall=gr6j_result["effective_rainfall"],
        q9=gr6j_result["q9"],
        q1=gr6j_result["q1"],
        routing_store=gr6j_result["routing_store"],
        exchange=gr6j_result["exchange"],
        actual_exchange_routing=gr6j_result["actual_exchange_routing"],
        actual_exchange_direct=gr6j_result["actual_exchange_direct"],
        actual_exchange_total=gr6j_result["actual_exchange_total"],
        qr=gr6j_result["qr"],
        qrexp=gr6j_result["qrexp"],
        exponential_store=gr6j_result["exponential_store"],
        qd=gr6j_result["qd"],
        streamflow=gr6j_result["streamflow"],
    )

    # Build SnowOutput for backward compatibility
    snow_output = SnowOutput(
        precip_raw=forcing.precip.copy(),
        snow_pliq=snow_result["snow_pliq"],
        snow_psol=snow_result["snow_psol"],
        snow_pack=snow_result["snow_pack"],
        snow_thermal_state=snow_result["snow_thermal_state"],
        snow_gratio=snow_result["snow_gratio"],
        snow_pot_melt=snow_result["snow_pot_melt"],
        snow_melt=snow_result["snow_melt"],
        snow_pliq_and_melt=snow_result["snow_pliq_and_melt"],
        snow_temp=snow_result["snow_temp"],
        snow_gthreshold=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
        snow_glocalmax=np.full(n_timesteps, catchment.mean_annual_solid_precip * 0.9),
    )

    # Build per-layer outputs if multi-layer
    snow_layers: SnowLayerOutputs | None = None
    if n_layers > 1:
        snow_layers = SnowLayerOutputs(
            layer_elevations=layer_elevations,
            layer_fractions=layer_fractions,
            snow_pack=layer_result["snow_pack"],
            snow_thermal_state=layer_result["snow_thermal_state"],
            snow_gratio=layer_result["snow_gratio"],
            snow_melt=layer_result["snow_melt"],
            snow_pliq_and_melt=layer_result["snow_pliq_and_melt"],
            layer_temp=layer_result["layer_temp"],
            layer_precip=layer_result["layer_precip"],
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=combined_fluxes,
        snow=snow_output,
        snow_layers=snow_layers,
    )
