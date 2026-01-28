"""Tests for the metrics registry."""

import logging

import pytest

from gr6j.calibration.metrics import (
    METRICS,
    get_metric,
    list_metrics,
    register,
    validate_objectives,
)


class TestRegistry:
    """Tests for metric registration and retrieval."""

    def test_metrics_populated(self) -> None:
        """Registry should contain registered metrics."""
        # Functions module registers 6 metrics
        assert len(METRICS) >= 6
        assert "nse" in METRICS
        assert "kge" in METRICS

    def test_get_metric_returns_function_and_direction(self) -> None:
        """get_metric returns (function, direction) tuple."""
        func, direction = get_metric("nse")
        assert callable(func)
        assert direction in ("maximize", "minimize")

    def test_get_metric_unknown_raises_keyerror(self) -> None:
        """get_metric raises KeyError for unknown metrics."""
        with pytest.raises(KeyError, match="Unknown metric 'nonexistent'"):
            get_metric("nonexistent")

    def test_list_metrics_returns_sorted_names(self) -> None:
        """list_metrics returns sorted list of metric names."""
        names = list_metrics()
        assert isinstance(names, list)
        assert names == sorted(names)
        assert "nse" in names
        assert "kge" in names

    def test_register_invalid_direction_raises(self) -> None:
        """register raises ValueError for invalid direction."""
        with pytest.raises(ValueError, match="direction must be"):
            register("invalid")


class TestValidateObjectives:
    """Tests for validate_objectives function."""

    def test_valid_objectives(self) -> None:
        """Valid objectives should pass without error."""
        validate_objectives({"nse": "maximize"})
        validate_objectives({"nse": "maximize", "rmse": "minimize"})

    def test_empty_objectives_raises(self) -> None:
        """Empty objectives dict should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_objectives({})

    def test_unknown_metric_raises(self) -> None:
        """Unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            validate_objectives({"unknown_metric": "maximize"})

    def test_wrong_direction_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Wrong direction should log a warning."""
        with caplog.at_level(logging.WARNING):
            validate_objectives({"nse": "minimize"})  # NSE is registered as maximize
        assert "unusual" in caplog.text.lower() or "direction" in caplog.text.lower()
