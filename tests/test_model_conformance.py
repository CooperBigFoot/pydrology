"""Parametrized conformance suite for all registered PyDrology models.

Verifies that all 4 registered models (gr2m, gr6j, hbv_light, gr6j_cemaneige)
follow the standard model contract: Parameters/State roundtrip, run() returns
ModelOutput with correct streamflow, non-negative streamflow, finite outputs,
registry consistency, and valid bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydrology import Catchment, ForcingData, Resolution, registry
from pydrology.outputs import ModelOutput

ALL_MODELS: list[str] = ["gr2m", "gr6j", "hbv_light", "gr6j_cemaneige"]


def _model_fixtures(model_name: str) -> tuple[object, ForcingData, dict[str, object]]:
    """Return (params, forcing, run_kwargs) for a given model.

    Computes midpoint parameters from the model's DEFAULT_BOUNDS and builds
    appropriate forcing data and keyword arguments for each model.

    Args:
        model_name: Registered model name.

    Returns:
        Tuple of (params, forcing, run_kwargs) ready for model.run().
    """
    model = registry.get_model(model_name)
    rng = np.random.default_rng(42)

    # Compute midpoint parameters from bounds
    bounds = model.DEFAULT_BOUNDS
    param_names = model.PARAM_NAMES
    midpoints = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in param_names])
    params = model.Parameters.from_array(midpoints)

    run_kwargs: dict[str, object] = {}

    if model_name == "gr2m":
        n = 12
        time = pd.date_range("2020-01-01", periods=n, freq="MS")
        forcing = ForcingData(
            time=time.to_numpy(),
            precip=rng.uniform(20.0, 120.0, size=n),
            pet=rng.uniform(10.0, 80.0, size=n),
            resolution=Resolution.monthly,
        )

    elif model_name == "gr6j":
        n = 10
        time = pd.date_range("2020-01-01", periods=n, freq="D")
        forcing = ForcingData(
            time=time.to_numpy(),
            precip=rng.uniform(0.0, 20.0, size=n),
            pet=rng.uniform(0.0, 5.0, size=n),
            resolution=Resolution.daily,
        )

    elif model_name == "hbv_light":
        n = 10
        time = pd.date_range("2020-01-01", periods=n, freq="D")
        forcing = ForcingData(
            time=time.to_numpy(),
            precip=rng.uniform(0.0, 20.0, size=n),
            pet=rng.uniform(0.0, 5.0, size=n),
            temp=rng.uniform(-5.0, 15.0, size=n),
            resolution=Resolution.daily,
        )

    elif model_name == "gr6j_cemaneige":
        n = 10
        time = pd.date_range("2020-01-01", periods=n, freq="D")
        forcing = ForcingData(
            time=time.to_numpy(),
            precip=rng.uniform(0.0, 20.0, size=n),
            pet=rng.uniform(0.0, 5.0, size=n),
            temp=rng.uniform(-5.0, 15.0, size=n),
            resolution=Resolution.daily,
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)
        run_kwargs["catchment"] = catchment

    else:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)

    return params, forcing, run_kwargs


def _make_initial_state(model_name: str, params: object) -> object:
    """Create an initial state for the given model.

    Handles model-specific initialization signatures:
    - GR2M, GR6J: State.initialize(params)
    - HBV-Light: State.initialize(params, n_zones=1)
    - GR6J-CemaNeige: State.initialize(params, catchment)

    Args:
        model_name: Registered model name.
        params: Parameters instance for the model.

    Returns:
        Initialized State instance.
    """
    model = registry.get_model(model_name)

    if model_name in ("gr2m", "gr6j"):
        return model.State.initialize(params)
    elif model_name == "hbv_light":
        return model.State.initialize(params, n_zones=1)
    elif model_name == "gr6j_cemaneige":
        catchment = Catchment(mean_annual_solid_precip=150.0)
        return model.State.initialize(params, catchment)
    else:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)


def _reconstruct_state(model_name: str, arr: np.ndarray) -> object:
    """Reconstruct a State from array for the given model.

    Handles model-specific from_array signatures:
    - GR2M, GR6J: State.from_array(arr)
    - HBV-Light: State.from_array(arr, n_zones=1)
    - GR6J-CemaNeige: State.from_array(arr, n_layers=1)

    Args:
        model_name: Registered model name.
        arr: 1D state array.

    Returns:
        Reconstructed State instance.
    """
    model = registry.get_model(model_name)

    if model_name in ("gr2m", "gr6j"):
        return model.State.from_array(arr)
    elif model_name == "hbv_light":
        return model.State.from_array(arr, n_zones=1)
    elif model_name == "gr6j_cemaneige":
        return model.State.from_array(arr, n_layers=1)
    else:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)


class TestModelConformance:
    """Parametrized conformance tests for all 4 registered models."""

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_parameters_roundtrip(self, model_name: str) -> None:
        """Parameters survives from_array -> __array__ -> from_array roundtrip."""
        model = registry.get_model(model_name)
        bounds = model.DEFAULT_BOUNDS
        param_names = model.PARAM_NAMES

        midpoints = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in param_names])
        params = model.Parameters.from_array(midpoints)

        arr = np.asarray(params)
        reconstructed = model.Parameters.from_array(arr)
        arr_reconstructed = np.asarray(reconstructed)

        np.testing.assert_allclose(arr, midpoints, rtol=1e-12)
        np.testing.assert_allclose(arr_reconstructed, midpoints, rtol=1e-12)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_state_roundtrip(self, model_name: str) -> None:
        """State survives initialize -> __array__ -> from_array roundtrip."""
        model = registry.get_model(model_name)
        bounds = model.DEFAULT_BOUNDS
        param_names = model.PARAM_NAMES

        midpoints = np.array([(bounds[p][0] + bounds[p][1]) / 2 for p in param_names])
        params = model.Parameters.from_array(midpoints)

        state = _make_initial_state(model_name, params)
        arr = np.asarray(state)
        reconstructed = _reconstruct_state(model_name, arr)
        arr_reconstructed = np.asarray(reconstructed)

        assert arr.shape == arr_reconstructed.shape
        np.testing.assert_allclose(arr_reconstructed, arr, rtol=1e-12)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_run_returns_model_output_with_streamflow(self, model_name: str) -> None:
        """run() returns a ModelOutput whose fluxes.streamflow is correct length."""
        model = registry.get_model(model_name)
        params, forcing, run_kwargs = _model_fixtures(model_name)

        result = model.run(params, forcing, **run_kwargs)

        assert isinstance(result, ModelOutput)
        streamflow = result.fluxes.streamflow
        assert isinstance(streamflow, np.ndarray)
        assert len(streamflow) == len(forcing)

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_streamflow_non_negative(self, model_name: str) -> None:
        """All streamflow values are >= 0."""
        model = registry.get_model(model_name)
        params, forcing, run_kwargs = _model_fixtures(model_name)

        result = model.run(params, forcing, **run_kwargs)
        streamflow = result.fluxes.streamflow

        assert np.all(streamflow >= 0), f"Negative streamflow found: min={streamflow.min()}"

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_all_outputs_finite(self, model_name: str) -> None:
        """All values in fluxes.to_dict() are finite (no NaN or Inf)."""
        model = registry.get_model(model_name)
        params, forcing, run_kwargs = _model_fixtures(model_name)

        result = model.run(params, forcing, **run_kwargs)
        flux_dict = result.fluxes.to_dict()

        for key, arr in flux_dict.items():
            assert np.all(np.isfinite(arr)), (
                f"Non-finite values in '{key}': NaN count={np.sum(np.isnan(arr))}, Inf count={np.sum(np.isinf(arr))}"
            )

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_registry_exports_consistency(self, model_name: str) -> None:
        """PARAM_NAMES, DEFAULT_BOUNDS, and STATE_SIZE are consistent."""
        model = registry.get_model(model_name)

        assert len(model.PARAM_NAMES) == len(model.DEFAULT_BOUNDS)
        assert set(model.PARAM_NAMES) == set(model.DEFAULT_BOUNDS.keys())
        assert isinstance(model.STATE_SIZE, int)
        assert model.STATE_SIZE > 0

    @pytest.mark.parametrize("model_name", ALL_MODELS)
    def test_bounds_validity(self, model_name: str) -> None:
        """Every parameter bound has lower < upper."""
        model = registry.get_model(model_name)

        for param_name, (lower, upper) in model.DEFAULT_BOUNDS.items():
            assert lower < upper, f"Invalid bounds for '{param_name}': lower={lower} >= upper={upper}"
