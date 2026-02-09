"""Tests for HBV-Light Rust backend equivalence.

Tests verify that the Rust-backed HBV-Light functions produce correct and
consistent results between step-by-step and full run execution.
"""

import numpy as np
import pandas as pd
import pytest
from pydrology import ForcingData
from pydrology.models.hbv_light import Parameters, State, compute_triangular_weights, run, step


class TestStepRunEquivalence:
    """Tests that step-by-step execution matches run()."""

    @pytest.fixture
    def params(self) -> Parameters:
        """Typical HBV-light parameters."""
        return Parameters(
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

    @pytest.fixture
    def forcing(self) -> ForcingData:
        """100-day daily forcing with random data."""
        n = 100
        rng = np.random.default_rng(42)
        return ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 30, n),
            pet=rng.uniform(1, 8, n),
            temp=rng.uniform(-5, 25, n),
        )

    def test_full_run_matches_step_loop(self, params: Parameters, forcing: ForcingData) -> None:
        """Full simulation produces identical streamflow to step-by-step."""
        result = run(params, forcing)

        uh_weights = compute_triangular_weights(params.maxbas)
        state = State.initialize(params)

        streamflow_step = []
        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
                float(forcing.temp[i]),
                uh_weights=uh_weights,
            )
            streamflow_step.append(fluxes["streamflow"])

        np.testing.assert_allclose(result.fluxes.streamflow, streamflow_step, rtol=1e-10)

    def test_all_outputs_match(self, params: Parameters, forcing: ForcingData) -> None:
        """All 20 HBV-light output fields match between run() and step loop."""
        result = run(params, forcing)

        uh_weights = compute_triangular_weights(params.maxbas)
        state = State.initialize(params)

        flux_keys = [
            "precip",
            "temp",
            "pet",
            "precip_rain",
            "precip_snow",
            "snow_pack",
            "snow_melt",
            "liquid_water_in_snow",
            "snow_input",
            "soil_moisture",
            "recharge",
            "actual_et",
            "upper_zone",
            "lower_zone",
            "q0",
            "q1",
            "q2",
            "percolation",
            "qgw",
            "streamflow",
        ]

        outputs_step: dict[str, list[float]] = {k: [] for k in flux_keys}

        for i in range(len(forcing)):
            state, fluxes = step(
                state,
                params,
                float(forcing.precip[i]),
                float(forcing.pet[i]),
                float(forcing.temp[i]),
                uh_weights=uh_weights,
            )
            for k in flux_keys:
                outputs_step[k].append(fluxes[k])

        for k in flux_keys:
            np.testing.assert_allclose(
                getattr(result.fluxes, k),
                outputs_step[k],
                rtol=1e-10,
                err_msg=f"Field {k} does not match",
            )


class TestEdgeCases:
    """Tests for edge cases in HBV-light implementation."""

    @pytest.fixture
    def params(self) -> Parameters:
        """Typical HBV-light parameters."""
        return Parameters(
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

    def test_zero_precipitation(self, params: Parameters) -> None:
        """Handles zero precipitation correctly."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.zeros(n),
            pet=np.full(n, 3.0),
            temp=np.full(n, 5.0),
        )
        result = run(params, forcing)

        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_zero_pet(self, params: Parameters) -> None:
        """Handles zero PET correctly."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 10.0),
            pet=np.zeros(n),
            temp=np.full(n, 5.0),
        )
        result = run(params, forcing)

        assert np.all(result.fluxes.streamflow >= 0)
        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_all_snow_conditions(self, params: Parameters) -> None:
        """Handles all-snow conditions correctly."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 10.0),
            pet=np.full(n, 3.0),
            temp=np.full(n, -10.0),
        )
        result = run(params, forcing)

        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_all_rain_conditions(self, params: Parameters) -> None:
        """Handles all-rain conditions correctly."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 10.0),
            pet=np.full(n, 3.0),
            temp=np.full(n, 20.0),
        )
        result = run(params, forcing)

        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.any(result.fluxes.streamflow > 0)


class TestNumericalStability:
    """Tests for numerical stability."""

    @pytest.fixture
    def params(self) -> Parameters:
        """Typical HBV-light parameters."""
        return Parameters(
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

    def test_extreme_precip(self, params: Parameters) -> None:
        """Handles extreme precipitation without blowing up."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 500.0),
            pet=np.full(n, 3.0),
            temp=np.full(n, 5.0),
        )
        result = run(params, forcing)

        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(result.fluxes.streamflow >= 0)

    def test_tiny_precip(self, params: Parameters) -> None:
        """Handles tiny precipitation without numerical issues."""
        n = 50
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=np.full(n, 0.001),
            pet=np.full(n, 3.0),
            temp=np.full(n, 5.0),
        )
        result = run(params, forcing)

        assert np.all(np.isfinite(result.fluxes.streamflow))

    def test_long_simulation(self, params: Parameters) -> None:
        """Long simulation maintains numerical stability."""
        n = 1000
        rng = np.random.default_rng(42)
        forcing = ForcingData(
            time=pd.date_range("2020-01-01", periods=n, freq="D").values,
            precip=rng.uniform(0, 30, n),
            pet=rng.uniform(1, 8, n),
            temp=rng.uniform(-5, 25, n),
        )
        result = run(params, forcing)

        assert np.all(np.isfinite(result.fluxes.streamflow))
        assert np.all(result.fluxes.streamflow >= 0)
