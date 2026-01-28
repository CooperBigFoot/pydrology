"""GR6J hydrological model.

A lumped conceptual rainfall-runoff model for daily streamflow simulation.
Génie Rural à 6 paramètres Journalier.

Includes optional CemaNeige snow module for cold-climate catchments.
"""

from .calibration import ObservedData, Solution, calibrate, list_metrics
from .cemaneige import (
    CemaNeige,
    CemaNeigeMultiLayerState,
    CemaNeigeSingleLayerState,
    cemaneige_multi_layer_step,
    cemaneige_step,
)
from .inputs import Catchment, ForcingData
from .model import Parameters, State, run, step
from .outputs import GR6JOutput, ModelOutput, SnowLayerOutputs, SnowOutput

__all__ = [
    "Catchment",
    "CemaNeige",
    "CemaNeigeMultiLayerState",
    "CemaNeigeSingleLayerState",
    "ForcingData",
    "GR6JOutput",
    "ModelOutput",
    "ObservedData",
    "Parameters",
    "SnowLayerOutputs",
    "SnowOutput",
    "Solution",
    "State",
    "calibrate",
    "cemaneige_multi_layer_step",
    "cemaneige_step",
    "list_metrics",
    "run",
    "step",
]
