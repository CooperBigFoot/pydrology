"""Tests for HBV-light process functions."""

import numpy as np
import pytest

from pydrology.processes.hbv_light import (
    compute_actual_et,
    compute_melt,
    compute_percolation,
    compute_recharge,
    compute_refreezing,
    compute_triangular_weights,
    convolve_triangular,
    lower_zone_outflow,
    partition_precipitation,
    upper_zone_outflows,
)


class TestPartitionPrecipitation:
    """Tests for precipitation partitioning."""

    def test_rain_above_threshold(self) -> None:
        """All precip is rain when temp > TT."""
        p_rain, p_snow = partition_precipitation(10.0, 5.0, 0.0, 1.0)
        assert p_rain == 10.0
        assert p_snow == 0.0

    def test_snow_below_threshold(self) -> None:
        """All precip is snow when temp <= TT."""
        p_rain, p_snow = partition_precipitation(10.0, -5.0, 0.0, 1.0)
        assert p_rain == 0.0
        assert p_snow == 10.0

    def test_sfcf_correction(self) -> None:
        """SFCF corrects snowfall."""
        p_rain, p_snow = partition_precipitation(10.0, -5.0, 0.0, 0.8)
        assert p_rain == 0.0
        assert p_snow == 8.0


class TestSnowMelt:
    """Tests for snowmelt computation."""

    def test_melt_above_threshold(self) -> None:
        """Melt occurs when temp > TT."""
        melt = compute_melt(5.0, 0.0, 3.0, 100.0)
        assert melt == pytest.approx(15.0)  # 3 * 5

    def test_no_melt_below_threshold(self) -> None:
        """No melt when temp <= TT."""
        melt = compute_melt(-5.0, 0.0, 3.0, 100.0)
        assert melt == 0.0

    def test_melt_limited_by_snow(self) -> None:
        """Melt cannot exceed available snow."""
        melt = compute_melt(10.0, 0.0, 3.0, 5.0)  # Would be 30mm but only 5mm snow
        assert melt == 5.0


class TestSnowRefreeze:
    """Tests for refreezing computation."""

    def test_refreeze_below_threshold(self) -> None:
        """Refreezing occurs when temp < TT."""
        refreeze = compute_refreezing(-5.0, 0.0, 3.0, 0.05, 10.0)
        assert refreeze == pytest.approx(0.75)  # 0.05 * 3 * 5

    def test_no_refreeze_above_threshold(self) -> None:
        """No refreezing when temp >= TT."""
        refreeze = compute_refreezing(5.0, 0.0, 3.0, 0.05, 10.0)
        assert refreeze == 0.0

    def test_refreeze_limited_by_liquid(self) -> None:
        """Refreezing limited by available liquid water."""
        refreeze = compute_refreezing(-10.0, 0.0, 3.0, 0.5, 2.0)  # Would be 15mm
        assert refreeze == 2.0


class TestSoilRecharge:
    """Tests for soil recharge computation."""

    def test_recharge_increases_with_moisture(self) -> None:
        """Higher soil moisture gives higher recharge."""
        low_sm = compute_recharge(10.0, 50.0, 250.0, 2.0)
        high_sm = compute_recharge(10.0, 200.0, 250.0, 2.0)
        assert high_sm > low_sm

    def test_recharge_at_saturation(self) -> None:
        """At saturation (SM=FC), all input becomes recharge."""
        recharge = compute_recharge(10.0, 250.0, 250.0, 2.0)
        assert recharge == pytest.approx(10.0)

    def test_no_recharge_dry_soil(self) -> None:
        """Very low soil moisture gives minimal recharge."""
        recharge = compute_recharge(10.0, 1.0, 250.0, 2.0)
        assert recharge < 0.01


class TestActualEvapotranspiration:
    """Tests for actual ET computation."""

    def test_full_et_above_lp_threshold(self) -> None:
        """Full PET when SM >= LP*FC."""
        et = compute_actual_et(5.0, 225.0, 250.0, 0.9)  # SM=225, LP*FC=225
        assert et == pytest.approx(5.0)

    def test_reduced_et_below_lp_threshold(self) -> None:
        """Reduced ET when SM < LP*FC."""
        et = compute_actual_et(5.0, 112.5, 250.0, 0.9)  # SM=112.5, LP*FC=225
        assert et == pytest.approx(2.5)  # Linear reduction

    def test_et_limited_by_available(self) -> None:
        """ET cannot exceed available soil moisture."""
        # When SM >= LP*FC, ET would be full PET, but limited by available SM
        # LP*FC = 0.9 * 250 = 225. Use SM=230 (above threshold) with high PET
        et = compute_actual_et(300.0, 230.0, 250.0, 0.9)
        assert et == pytest.approx(230.0)  # Limited by available SM


class TestUpperZoneOutflow:
    """Tests for upper zone outflows."""

    def test_q0_above_uzl(self) -> None:
        """Q0 activates when SUZ > UZL."""
        q0, q1 = upper_zone_outflows(50.0, 0.4, 0.1, 20.0)
        assert q0 == pytest.approx(12.0)  # 0.4 * (50-20)
        assert q1 == pytest.approx(5.0)  # 0.1 * 50

    def test_no_q0_below_uzl(self) -> None:
        """No Q0 when SUZ <= UZL."""
        q0, q1 = upper_zone_outflows(15.0, 0.4, 0.1, 20.0)
        assert q0 == 0.0
        assert q1 == pytest.approx(1.5)  # 0.1 * 15


class TestPercolation:
    """Tests for percolation computation."""

    def test_percolation_limited_by_max(self) -> None:
        """Percolation is limited by PERC parameter."""
        perc = compute_percolation(100.0, 2.0)
        assert perc == 2.0

    def test_percolation_limited_by_storage(self) -> None:
        """Percolation is limited by available storage."""
        perc = compute_percolation(1.0, 2.0)
        assert perc == 1.0


class TestLowerZoneOutflow:
    """Tests for lower zone outflow."""

    def test_baseflow_linear(self) -> None:
        """Baseflow is linear with storage."""
        q2 = lower_zone_outflow(100.0, 0.01)
        assert q2 == pytest.approx(1.0)


class TestTriangularUhWeights:
    """Tests for triangular unit hydrograph weights."""

    def test_weights_sum_to_one(self) -> None:
        """Weights should sum to 1.0."""
        for maxbas in [1.0, 2.0, 2.5, 3.0, 5.5, 7.0]:
            weights = compute_triangular_weights(maxbas)
            assert weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_weights_length(self) -> None:
        """Weights length is ceil(MAXBAS)."""
        assert len(compute_triangular_weights(1.0)) == 1
        assert len(compute_triangular_weights(2.0)) == 2
        assert len(compute_triangular_weights(2.5)) == 3
        assert len(compute_triangular_weights(7.0)) == 7

    def test_integer_maxbas_symmetric(self) -> None:
        """Integer MAXBAS gives symmetric weights."""
        weights = compute_triangular_weights(4.0)
        assert weights[0] == pytest.approx(weights[3], abs=1e-10)
        assert weights[1] == pytest.approx(weights[2], abs=1e-10)


class TestConvolveTriangularUh:
    """Tests for triangular UH convolution."""

    def test_convolution_conserves_mass(self) -> None:
        """Total input equals total output over time."""
        weights = compute_triangular_weights(3.0)
        buffer = np.zeros(7)

        total_input = 0.0
        total_output = 0.0

        # Add impulse
        buffer, qsim = convolve_triangular(100.0, buffer, weights)
        total_input += 100.0
        total_output += qsim

        # Run for enough timesteps to flush buffer
        for _ in range(10):
            buffer, qsim = convolve_triangular(0.0, buffer, weights)
            total_output += qsim

        assert total_output == pytest.approx(total_input, abs=1e-10)
