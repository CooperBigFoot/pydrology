"""HBV-light model orchestration functions.

This module provides the main entry points for running the HBV-light model:
- step(): Execute a single timestep
- run(): Execute the model over a timeseries
"""

import numpy as np

from pydrology._core import hbv_light as _rust
from pydrology.outputs import ModelOutput
from pydrology.types import Catchment, ForcingData
from pydrology.utils.elevation import (
    ELEV_CAP_PRECIP,
    GRAD_P_DEFAULT,
    GRAD_T_DEFAULT,
    derive_layers,
)
from .outputs import HBVLightFluxes, HBVLightZoneOutputs
from .routing import compute_triangular_weights
from .constants import SUPPORTED_RESOLUTIONS
from .types import Parameters, State


def step(
    state: State,
    params: Parameters,
    precip: float,
    pet: float,
    temp: float,
    catchment: Catchment | None = None,
    uh_weights: np.ndarray | None = None,
    zone_elevations: np.ndarray | None = None,
    zone_fractions: np.ndarray | None = None,
    input_elevation: float | None = None,
    temp_gradient: float | None = None,
    precip_gradient: float | None = None,
) -> tuple[State, dict[str, float]]:
    """Execute one timestep of the HBV-light model.

    Args:
        state: Current model state (stores and routing buffer).
        params: Model parameters.
        precip: Daily precipitation [mm/day].
        pet: Daily potential evapotranspiration [mm/day].
        temp: Daily mean temperature [C].
        catchment: Optional catchment properties (unused, for API compatibility).
        uh_weights: Pre-computed UH weights. If None, computed from params.maxbas.
        zone_elevations: Representative elevation of each zone [m]. Shape (n_zones,).
        zone_fractions: Area fraction of each zone [-]. Shape (n_zones,).
        input_elevation: Elevation of input measurement [m]. Required for extrapolation.
        temp_gradient: Temperature lapse rate [C/100m]. Default is GRAD_T_DEFAULT.
        precip_gradient: Precipitation gradient [m^-1]. Default is GRAD_P_DEFAULT.

    Returns:
        Tuple of (new_state, fluxes) where:
        - new_state: Updated State object after the timestep
        - fluxes: Dictionary containing all model outputs (area-weighted aggregates)
    """
    if uh_weights is None:
        uh_weights = compute_triangular_weights(params.maxbas)

    state_arr = np.asarray(state, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)

    new_state_arr, fluxes = _rust.hbv_step(
        state_arr,
        params_arr,
        precip,
        pet,
        temp,
        uh_weights,
    )

    new_state = State.from_array(np.asarray(new_state_arr), state.n_zones)

    return new_state, fluxes


def run(
    params: Parameters,
    forcing: ForcingData,
    initial_state: State | None = None,
    catchment: Catchment | None = None,
) -> ModelOutput[HBVLightFluxes]:
    """Run the HBV-light model over a timeseries.

    Args:
        params: Model parameters.
        forcing: Input forcing data with precip, pet, and temp arrays.
        initial_state: Initial model state. If None, uses State.initialize(params).
        catchment: Optional catchment properties for elevation band support.

    Returns:
        ModelOutput containing HBVLightFluxes outputs.

    Raises:
        ValueError: If forcing.temp is None.
    """
    if forcing.temp is None:
        raise ValueError("HBV-light requires temperature data (forcing.temp)")

    if forcing.resolution not in SUPPORTED_RESOLUTIONS:
        supported = [r.value for r in SUPPORTED_RESOLUTIONS]
        msg = f"HBV-light supports resolutions {supported}, got '{forcing.resolution.value}'"
        raise ValueError(msg)

    # Determine zone configuration
    if catchment is not None and catchment.n_layers > 1 and catchment.hypsometric_curve is not None:
        n_zones = catchment.n_layers
        zone_elevations, zone_fractions = derive_layers(catchment.hypsometric_curve, n_zones)
        input_elevation = catchment.input_elevation
        temp_gradient = catchment.temp_gradient if catchment.temp_gradient is not None else GRAD_T_DEFAULT
        precip_gradient = catchment.precip_gradient if catchment.precip_gradient is not None else GRAD_P_DEFAULT
    else:
        n_zones = 1
        zone_elevations = None
        zone_fractions = None
        input_elevation = None
        temp_gradient = None
        precip_gradient = None

    # Prepare arrays
    params_arr = np.asarray(params, dtype=np.float64)
    precip_arr = forcing.precip.astype(np.float64)
    pet_arr = forcing.pet.astype(np.float64)
    temp_arr = forcing.temp.astype(np.float64)

    # Initial state
    state_arr = None
    if initial_state is not None:
        state_arr = np.asarray(initial_state, dtype=np.float64)

    # Call Rust backend
    fluxes_dict, zone_dict = _rust.hbv_run(
        params_arr,
        precip_arr,
        pet_arr,
        temp_arr,
        initial_state=state_arr,
        n_zones=n_zones,
        zone_elevations=zone_elevations.astype(np.float64) if zone_elevations is not None else None,
        zone_fractions=zone_fractions.astype(np.float64) if zone_fractions is not None else None,
        input_elevation=input_elevation,
        temp_gradient=temp_gradient,
        precip_gradient=precip_gradient,
    )

    # Build output object
    hbv_fluxes = HBVLightFluxes(
        precip=np.asarray(fluxes_dict["precip"]),
        temp=np.asarray(fluxes_dict["temp"]),
        pet=np.asarray(fluxes_dict["pet"]),
        precip_rain=np.asarray(fluxes_dict["precip_rain"]),
        precip_snow=np.asarray(fluxes_dict["precip_snow"]),
        snow_pack=np.asarray(fluxes_dict["snow_pack"]),
        snow_melt=np.asarray(fluxes_dict["snow_melt"]),
        liquid_water_in_snow=np.asarray(fluxes_dict["liquid_water_in_snow"]),
        snow_input=np.asarray(fluxes_dict["snow_input"]),
        soil_moisture=np.asarray(fluxes_dict["soil_moisture"]),
        recharge=np.asarray(fluxes_dict["recharge"]),
        actual_et=np.asarray(fluxes_dict["actual_et"]),
        upper_zone=np.asarray(fluxes_dict["upper_zone"]),
        lower_zone=np.asarray(fluxes_dict["lower_zone"]),
        q0=np.asarray(fluxes_dict["q0"]),
        q1=np.asarray(fluxes_dict["q1"]),
        q2=np.asarray(fluxes_dict["q2"]),
        percolation=np.asarray(fluxes_dict["percolation"]),
        qgw=np.asarray(fluxes_dict["qgw"]),
        streamflow=np.asarray(fluxes_dict["streamflow"]),
    )

    # Build per-zone outputs if multi-zone
    zone_outputs: HBVLightZoneOutputs | None = None
    if zone_dict is not None:
        n_t = int(zone_dict["n_timesteps"])
        n_z = int(zone_dict["n_zones"])
        zone_outputs = HBVLightZoneOutputs(
            zone_elevations=np.asarray(zone_dict["zone_elevations"]),
            zone_fractions=np.asarray(zone_dict["zone_fractions"]),
            zone_temp=np.asarray(zone_dict["zone_temp"]).reshape(n_t, n_z),
            zone_precip=np.asarray(zone_dict["zone_precip"]).reshape(n_t, n_z),
            snow_pack=np.asarray(zone_dict["snow_pack"]).reshape(n_t, n_z),
            liquid_water_in_snow=np.asarray(zone_dict["liquid_water_in_snow"]).reshape(n_t, n_z),
            snow_melt=np.asarray(zone_dict["snow_melt"]).reshape(n_t, n_z),
            snow_input=np.asarray(zone_dict["snow_input"]).reshape(n_t, n_z),
            soil_moisture=np.asarray(zone_dict["soil_moisture"]).reshape(n_t, n_z),
            recharge=np.asarray(zone_dict["recharge"]).reshape(n_t, n_z),
            actual_et=np.asarray(zone_dict["actual_et"]).reshape(n_t, n_z),
        )

    return ModelOutput(
        time=forcing.time,
        fluxes=hbv_fluxes,
        snow=None,
        snow_layers=None,
        zone_outputs=zone_outputs,
    )
