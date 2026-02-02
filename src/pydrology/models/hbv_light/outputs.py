"""HBV-light model flux outputs as arrays.

This module provides dataclasses for organizing and accessing HBV-light model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class HBVLightFluxes:
    """HBV-light model flux outputs as arrays.

    All arrays have the same length as the input forcing data.

    Attributes:
        precip: Precipitation input [mm/day].
        temp: Temperature input [°C].
        pet: Potential evapotranspiration [mm/day].
        precip_rain: Liquid precipitation [mm/day].
        precip_snow: Solid precipitation after SFCF correction [mm/day].
        snow_pack: Snow pack water equivalent after timestep [mm].
        snow_melt: Snowmelt [mm/day].
        liquid_water_in_snow: Liquid water held in snow pack [mm].
        snow_input: Total input to soil (rain + melt outflow) [mm/day].
        soil_moisture: Soil moisture after timestep [mm].
        recharge: Recharge to groundwater [mm/day].
        actual_et: Actual evapotranspiration [mm/day].
        upper_zone: Upper groundwater zone storage after timestep [mm].
        lower_zone: Lower groundwater zone storage after timestep [mm].
        q0: Surface/quick flow from upper zone [mm/day].
        q1: Interflow from upper zone [mm/day].
        q2: Baseflow from lower zone [mm/day].
        percolation: Percolation from upper to lower zone [mm/day].
        qgw: Total groundwater runoff before routing [mm/day].
        streamflow: Total simulated streamflow after routing [mm/day].
    """

    # Inputs
    precip: np.ndarray
    temp: np.ndarray
    pet: np.ndarray

    # Snow routine outputs
    precip_rain: np.ndarray
    precip_snow: np.ndarray
    snow_pack: np.ndarray
    snow_melt: np.ndarray
    liquid_water_in_snow: np.ndarray
    snow_input: np.ndarray

    # Soil routine outputs
    soil_moisture: np.ndarray
    recharge: np.ndarray
    actual_et: np.ndarray

    # Response routine outputs
    upper_zone: np.ndarray
    lower_zone: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    percolation: np.ndarray
    qgw: np.ndarray

    # Final output
    streamflow: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays.

        Returns:
            Dictionary mapping field names to their numpy array values.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}
