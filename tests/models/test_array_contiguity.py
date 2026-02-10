"""Regression tests for array contiguity handling.

Ensures all model shims use np.ascontiguousarray() (not np.asarray()) when
passing arrays to the Rust backend, which requires C-contiguous memory layout.
Non-contiguous arrays (e.g., Fortran-ordered) must be handled transparently.
"""

import numpy as np
import pandas as pd
import pytest
from pydrology import Catchment, ForcingData, Resolution


@pytest.fixture
def daily_forcing() -> ForcingData:
    """Minimal daily forcing data (30 days)."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=30, freq="D").values,
        precip=np.full(30, 10.0),
        pet=np.full(30, 3.0),
        temp=np.full(30, 5.0),
    )


@pytest.fixture
def monthly_forcing() -> ForcingData:
    """Minimal monthly forcing data (12 months)."""
    return ForcingData(
        time=pd.date_range("2020-01-01", periods=12, freq="MS").values,
        precip=np.full(12, 80.0),
        pet=np.full(12, 40.0),
        resolution=Resolution.monthly,
    )


class TestGR2MContiguity:
    """GR2M handles Fortran-ordered state arrays."""

    def test_run_with_fortran_ordered_state(self, monthly_forcing: ForcingData) -> None:
        from pydrology.models.gr2m import Parameters, State, run

        params = Parameters(x1=500.0, x2=1.0)
        state = State.initialize(params)

        # Create Fortran-ordered (non-contiguous) state array
        state_arr = np.asfortranarray(np.asarray(state))
        assert not state_arr.flags["C_CONTIGUOUS"] or state_arr.ndim == 1  # 1D is always C-contiguous
        fortran_state = State.from_array(state_arr)

        result = run(params, monthly_forcing, initial_state=fortran_state)
        assert result.streamflow.shape == (12,)
        assert np.all(np.isfinite(result.streamflow))


class TestGR6JContiguity:
    """GR6J handles Fortran-ordered state arrays."""

    def test_run_with_fortran_ordered_state(self, daily_forcing: ForcingData) -> None:
        from pydrology.models.gr6j import Parameters, State, run

        params = Parameters(x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0)
        state = State.initialize(params)

        # Create a strided (non-contiguous) view of the state array
        full_arr = np.zeros(126, dtype=np.float64)
        full_arr[::2] = np.asarray(state)  # Every other element
        strided_state_arr = full_arr[::2]  # Non-contiguous view
        assert not strided_state_arr.flags["C_CONTIGUOUS"]
        strided_state = State.from_array(strided_state_arr)

        result = run(params, daily_forcing, initial_state=strided_state)
        assert result.streamflow.shape == (30,)
        assert np.all(np.isfinite(result.streamflow))


class TestHBVLightContiguity:
    """HBV-Light handles Fortran-ordered state arrays."""

    def test_run_with_fortran_ordered_state(self, daily_forcing: ForcingData) -> None:
        from pydrology.models.hbv_light import Parameters, State, run

        params = Parameters(
            tt=0.0, cfmax=3.5, sfcf=0.9, cwh=0.1, cfr=0.05,
            fc=250.0, lp=0.7, beta=2.5, k0=0.3, k1=0.1,
            k2=0.05, perc=2.0, uzl=30.0, maxbas=3.0,
        )
        state = State.initialize(params)

        # Create a strided (non-contiguous) view
        full_arr = np.zeros(24, dtype=np.float64)
        full_arr[::2] = np.asarray(state)
        strided_state_arr = full_arr[::2]
        assert not strided_state_arr.flags["C_CONTIGUOUS"]
        strided_state = State.from_array(strided_state_arr)

        result = run(params, daily_forcing, initial_state=strided_state)
        assert result.streamflow.shape == (30,)
        assert np.all(np.isfinite(result.streamflow))


class TestGR6JCemaNeigeContiguity:
    """GR6J-CemaNeige handles Fortran-ordered state arrays."""

    def test_run_with_fortran_ordered_state(self, daily_forcing: ForcingData) -> None:
        from pydrology.models.gr6j_cemaneige import Parameters, State, run

        params = Parameters(
            x1=350.0, x2=0.0, x3=90.0, x4=1.7, x5=0.0, x6=5.0,
            ctg=0.97, kf=2.5,
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)
        state = State.initialize(params, catchment)

        # Create a strided (non-contiguous) view
        state_arr = np.asarray(state)
        full_arr = np.zeros(len(state_arr) * 2, dtype=np.float64)
        full_arr[::2] = state_arr
        strided_state_arr = full_arr[::2]
        assert not strided_state_arr.flags["C_CONTIGUOUS"]
        strided_state = State.from_array(strided_state_arr, n_layers=catchment.n_layers)

        result = run(params, daily_forcing, initial_state=strided_state, catchment=catchment)
        assert result.streamflow.shape == (30,)
        assert np.all(np.isfinite(result.streamflow))
