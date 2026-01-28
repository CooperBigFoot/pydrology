"""Tests for the calibrate function."""

import numpy as np
import pytest

from gr6j import Catchment, ForcingData, Parameters
from gr6j.calibration import ObservedData, Solution, calibrate
from gr6j.calibration.calibrate import (
    _array_to_parameters,
    _validate_bounds,
    _validate_snow_config,
    _validate_warmup,
)


class TestValidateBounds:
    """Tests for _validate_bounds helper."""

    def test_empty_bounds_raises(self) -> None:
        """Empty bounds should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_bounds({}, snow=False)

    def test_missing_gr6j_params_raises(self) -> None:
        """Missing GR6J parameters should raise ValueError."""
        bounds = {"x1": (1, 2500), "x2": (-5, 5)}  # Missing x3-x6
        with pytest.raises(ValueError, match="Missing bounds"):
            _validate_bounds(bounds, snow=False)

    def test_missing_snow_params_raises(self) -> None:
        """Missing snow parameters when snow=True should raise ValueError."""
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }  # Missing ctg, kf
        with pytest.raises(ValueError, match="Missing bounds"):
            _validate_bounds(bounds, snow=True)

    def test_lower_ge_upper_raises(self) -> None:
        """Lower bound >= upper bound should raise ValueError."""
        bounds = {
            "x1": (2500, 1),  # Invalid: lower > upper
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }
        with pytest.raises(ValueError, match="Lower bound must be less than upper"):
            _validate_bounds(bounds, snow=False)

    def test_valid_bounds_no_snow(self) -> None:
        """Valid GR6J bounds should pass."""
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
        }
        _validate_bounds(bounds, snow=False)  # Should not raise

    def test_valid_bounds_with_snow(self) -> None:
        """Valid GR6J + snow bounds should pass."""
        bounds = {
            "x1": (1, 2500),
            "x2": (-5, 5),
            "x3": (1, 1000),
            "x4": (0.5, 10),
            "x5": (-4, 4),
            "x6": (0.01, 20),
            "ctg": (0, 1),
            "kf": (0, 200),
        }
        _validate_bounds(bounds, snow=True)  # Should not raise


class TestValidateSnowConfig:
    """Tests for _validate_snow_config helper."""

    def test_no_snow_always_valid(self) -> None:
        """When snow=False, any config should pass."""
        forcing = ForcingData(
            time=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
            precip=np.array([10.0, 5.0]),
            pet=np.array([3.0, 4.0]),
        )
        _validate_snow_config(snow=False, forcing=forcing, catchment=None)

    def test_snow_without_temp_raises(self) -> None:
        """snow=True without temp should raise ValueError."""
        forcing = ForcingData(
            time=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
            precip=np.array([10.0, 5.0]),
            pet=np.array([3.0, 4.0]),
            temp=None,
        )
        with pytest.raises(ValueError, match="temp required"):
            _validate_snow_config(snow=True, forcing=forcing, catchment=None)

    def test_snow_without_catchment_raises(self) -> None:
        """snow=True without catchment should raise ValueError."""
        forcing = ForcingData(
            time=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64"),
            precip=np.array([10.0, 5.0]),
            pet=np.array([3.0, 4.0]),
            temp=np.array([5.0, 6.0]),
        )
        with pytest.raises(ValueError, match="catchment required"):
            _validate_snow_config(snow=True, forcing=forcing, catchment=None)


class TestValidateWarmup:
    """Tests for _validate_warmup helper."""

    def test_negative_warmup_raises(self) -> None:
        """Negative warmup should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            _validate_warmup(warmup=-1, forcing_length=100, observed_length=100)

    def test_length_mismatch_raises(self) -> None:
        """Observed length != forcing - warmup should raise ValueError."""
        with pytest.raises(ValueError, match="must equal"):
            _validate_warmup(warmup=10, forcing_length=100, observed_length=80)  # Should be 90

    def test_valid_warmup(self) -> None:
        """Valid warmup should pass."""
        _validate_warmup(warmup=10, forcing_length=100, observed_length=90)


class TestArrayToParameters:
    """Tests for _array_to_parameters helper."""

    def test_no_snow(self) -> None:
        """Without snow, should create Parameters with 6 values."""
        x = np.array([350.0, 0.0, 90.0, 1.7, 0.0, 5.0])
        params = _array_to_parameters(x, snow=False)
        assert params.x1 == 350.0
        assert params.x2 == 0.0
        assert params.x3 == 90.0
        assert params.x4 == 1.7
        assert params.x5 == 0.0
        assert params.x6 == 5.0
        assert params.snow is None

    def test_with_snow(self) -> None:
        """With snow, should create Parameters with CemaNeige."""
        x = np.array([350.0, 0.0, 90.0, 1.7, 0.0, 5.0, 0.97, 2.5])
        params = _array_to_parameters(x, snow=True)
        assert params.x1 == 350.0
        assert params.snow is not None
        assert params.snow.ctg == 0.97
        assert params.snow.kf == 2.5


class TestCalibrate:
    """Integration tests for calibrate function."""

    @pytest.fixture
    def simple_forcing(self) -> ForcingData:
        """Create simple forcing data for testing."""
        n_days = 30  # Short for fast tests
        return ForcingData(
            time=np.datetime64("2020-01-01") + np.arange(n_days),
            precip=np.random.default_rng(42).exponential(5.0, n_days),
            pet=np.full(n_days, 3.5),
        )

    @pytest.fixture
    def simple_observed(self, simple_forcing: ForcingData) -> ObservedData:
        """Create observed data matching forcing minus warmup."""
        warmup = 10
        n_obs = len(simple_forcing) - warmup
        return ObservedData(
            time=simple_forcing.time[warmup:],
            streamflow=np.random.default_rng(42).uniform(1.0, 5.0, n_obs),
        )

    @pytest.fixture
    def simple_bounds(self) -> dict[str, tuple[float, float]]:
        """Simple parameter bounds."""
        return {
            "x1": (100, 500),
            "x2": (-2, 2),
            "x3": (50, 200),
            "x4": (1, 5),
            "x5": (-2, 2),
            "x6": (1, 10),
        }

    def test_single_objective_returns_solution(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Single objective should return a Solution."""
        result = calibrate(
            forcing=simple_forcing,
            observed=simple_observed,
            objectives={"nse": "maximize"},
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, Solution)
        assert "nse" in result.score
        assert isinstance(result.parameters, Parameters)

    def test_multi_objective_returns_list(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Multiple objectives should return a list of Solutions."""
        result = calibrate(
            forcing=simple_forcing,
            observed=simple_observed,
            objectives={"nse": "maximize", "log_nse": "maximize"},
            bounds=simple_bounds,
            warmup=10,
            population_size=10,
            generations=3,
            seed=42,
        )
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(s, Solution) for s in result)
        assert all("nse" in s.score and "log_nse" in s.score for s in result)

    def test_deterministic_with_seed(
        self,
        simple_forcing: ForcingData,
        simple_observed: ObservedData,
        simple_bounds: dict[str, tuple[float, float]],
    ) -> None:
        """Same seed should produce same results."""
        kwargs = {
            "forcing": simple_forcing,
            "observed": simple_observed,
            "objectives": {"nse": "maximize"},
            "bounds": simple_bounds,
            "warmup": 10,
            "population_size": 10,
            "generations": 3,
            "seed": 12345,
        }
        result1 = calibrate(**kwargs)
        result2 = calibrate(**kwargs)
        assert result1.parameters.x1 == result2.parameters.x1
        assert result1.score["nse"] == result2.score["nse"]

    def test_snow_calibration(self) -> None:
        """Calibration with snow should work."""
        n_days = 30
        forcing = ForcingData(
            time=np.datetime64("2020-01-01") + np.arange(n_days),
            precip=np.random.default_rng(42).exponential(5.0, n_days),
            pet=np.full(n_days, 3.5),
            temp=np.random.default_rng(42).uniform(-5.0, 10.0, n_days),
        )
        catchment = Catchment(mean_annual_solid_precip=150.0)
        warmup = 10
        observed = ObservedData(
            time=forcing.time[warmup:],
            streamflow=np.random.default_rng(42).uniform(1.0, 5.0, n_days - warmup),
        )
        bounds = {
            "x1": (100, 500),
            "x2": (-2, 2),
            "x3": (50, 200),
            "x4": (1, 5),
            "x5": (-2, 2),
            "x6": (1, 10),
            "ctg": (0.5, 1.0),
            "kf": (1.0, 10.0),
        }

        result = calibrate(
            forcing=forcing,
            observed=observed,
            objectives={"nse": "maximize"},
            bounds=bounds,
            catchment=catchment,
            snow=True,
            warmup=warmup,
            population_size=10,
            generations=3,
            seed=42,
        )

        assert isinstance(result, Solution)
        assert result.parameters.snow is not None
        assert 0.5 <= result.parameters.snow.ctg <= 1.0
        assert 1.0 <= result.parameters.snow.kf <= 10.0
