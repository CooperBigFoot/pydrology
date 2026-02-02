"""Integration tests for HBV-light registry and calibration."""

import numpy as np
import pandas as pd

from pydrology import (
    ForcingData,
    ObservedData,
    calibrate,
    get_model,
    get_model_info,
    list_models,
)


class TestRegistry:
    """Tests for model registry integration."""

    def test_model_in_list(self) -> None:
        """hbv_light should be in list_models()."""
        models = list_models()
        assert "hbv_light" in models

    def test_get_model_info(self) -> None:
        """get_model_info returns correct metadata."""
        info = get_model_info("hbv_light")

        assert len(info["param_names"]) == 14
        assert "tt" in info["param_names"]
        assert "maxbas" in info["param_names"]

        assert len(info["default_bounds"]) == 14
        assert "tt" in info["default_bounds"]

        assert info["state_size"] == 12  # For single zone

    def test_get_model_exports(self) -> None:
        """get_model returns module with required exports."""
        model = get_model("hbv_light")

        assert hasattr(model, "PARAM_NAMES")
        assert hasattr(model, "DEFAULT_BOUNDS")
        assert hasattr(model, "STATE_SIZE")
        assert hasattr(model, "Parameters")
        assert hasattr(model, "State")
        assert hasattr(model, "run")
        assert hasattr(model, "step")

    def test_run_via_registry(self) -> None:
        """Can run model via registry."""
        model = get_model("hbv_light")

        params = model.Parameters(
            tt=0.0,
            cfmax=3.0,
            sfcf=1.0,
            cwh=0.1,
            cfr=0.05,
            fc=250.0,
            lp=0.9,
            beta=2.0,
            k0=0.4,
            k1=0.1,
            k2=0.01,
            perc=1.0,
            uzl=20.0,
            maxbas=2.5,
        )

        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=10, freq="D").values,
            precip=np.random.default_rng(42).uniform(0, 20, 10),
            pet=np.random.default_rng(42).uniform(1, 6, 10),
            temp=np.random.default_rng(42).uniform(-5, 15, 10),
        )

        result = model.run(params, forcing)

        assert len(result) == 10
        assert hasattr(result, "streamflow")


class TestCalibration:
    """Tests for calibration integration."""

    def test_calibration_runs(self) -> None:
        """Calibration runs without error for hbv_light."""
        # Create synthetic data - need enough days for warmup
        warmup = 365
        n_obs = 50  # Days of observed data
        n_forcing = warmup + n_obs
        rng = np.random.default_rng(42)

        forcing_time = pd.date_range("2020-01-01", periods=n_forcing, freq="D").values
        forcing = ForcingData(
            time=forcing_time,
            precip=rng.uniform(0, 20, n_forcing),
            pet=rng.uniform(2, 5, n_forcing),
            temp=rng.uniform(0, 15, n_forcing),
        )

        # Observed data covers only post-warmup period
        observed = ObservedData(
            time=forcing_time[warmup:],
            streamflow=rng.uniform(1, 10, n_obs),
        )

        # Run calibration with minimal iterations
        result = calibrate(
            model="hbv_light",
            forcing=forcing,
            observed=observed,
            objectives=["nse"],
            use_default_bounds=True,
            warmup=warmup,
            generations=5,  # Minimal for speed
            progress=False,
        )

        assert result is not None
        assert hasattr(result, "parameters")
        assert hasattr(result, "score")

    def test_calibration_result_structure(self) -> None:
        """Calibration result has correct structure."""
        warmup = 365
        n_obs = 30
        n_forcing = warmup + n_obs
        rng = np.random.default_rng(42)

        forcing_time = pd.date_range("2020-01-01", periods=n_forcing, freq="D").values
        forcing = ForcingData(
            time=forcing_time,
            precip=rng.uniform(0, 15, n_forcing),
            pet=rng.uniform(2, 4, n_forcing),
            temp=rng.uniform(5, 20, n_forcing),
        )

        observed = ObservedData(
            time=forcing_time[warmup:],
            streamflow=rng.uniform(0.5, 5, n_obs),
        )

        result = calibrate(
            model="hbv_light",
            forcing=forcing,
            observed=observed,
            objectives=["nse"],
            use_default_bounds=True,
            warmup=warmup,
            generations=3,
            progress=False,
        )

        # Check result has the correct model
        assert result.model == "hbv_light"

        # Convert to array and check length
        params_arr = np.asarray(result.parameters)
        assert len(params_arr) == 14

        # Check parameters are within bounds
        info = get_model_info("hbv_light")
        for i, name in enumerate(info["param_names"]):
            param_value = params_arr[i]
            low, high = info["default_bounds"][name]
            assert low <= param_value <= high, f"Parameter {name} out of bounds"
