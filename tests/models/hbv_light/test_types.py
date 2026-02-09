"""Tests for HBV-light data types: Parameters and State."""

import numpy as np
import pytest

from pydrology.models.hbv_light import Parameters, State


class TestParameters:
    """Tests for the Parameters frozen dataclass."""

    def test_creates_with_valid_parameters(self) -> None:
        """Parameters instantiates correctly with typical values."""
        params = Parameters(
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
        assert params.tt == 0.0
        assert params.fc == 250.0
        assert params.maxbas == 2.5

    def test_is_frozen_dataclass(self) -> None:
        """Parameters is immutable."""
        params = Parameters(
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
        with pytest.raises(AttributeError):
            params.tt = 1.0  # type: ignore[misc]

    def test_to_array_shape(self) -> None:
        """Parameters.__array__ returns array of length 14."""
        params = Parameters(
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
        arr = np.asarray(params)
        assert arr.shape == (14,)

    def test_array_roundtrip(self) -> None:
        """Parameters can be reconstructed from array."""
        original = Parameters(
            tt=-1.0,
            cfmax=5.0,
            sfcf=0.8,
            cwh=0.15,
            cfr=0.1,
            fc=300.0,
            lp=0.7,
            beta=3.0,
            k0=0.3,
            k1=0.2,
            k2=0.05,
            perc=2.0,
            uzl=30.0,
            maxbas=3.5,
        )
        arr = np.asarray(original)
        restored = Parameters.from_array(arr)

        assert restored.tt == original.tt
        assert restored.cfmax == original.cfmax
        assert restored.fc == original.fc
        assert restored.maxbas == original.maxbas


class TestState:
    """Tests for the State mutable dataclass."""

    @pytest.fixture
    def valid_params(self) -> Parameters:
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

    def test_initialize_creates_correct_values(self, valid_params: Parameters) -> None:
        """State.initialize sets stores to expected values."""
        state = State.initialize(valid_params)

        # Snow pack and liquid water start at zero
        assert state.zone_states[0, 0] == 0.0  # snow pack
        assert state.zone_states[0, 1] == 0.0  # liquid water
        # Soil moisture at 50% of FC
        assert state.zone_states[0, 2] == pytest.approx(0.5 * 250.0)
        # Groundwater zones at zero
        assert state.upper_zone == 0.0
        assert state.lower_zone == 0.0

    def test_initialize_multi_zone(self, valid_params: Parameters) -> None:
        """State.initialize with multiple zones."""
        state = State.initialize(valid_params, n_zones=3)

        assert state.n_zones == 3
        assert state.zone_states.shape == (3, 3)
        # All zones have SM at 50% FC
        for i in range(3):
            assert state.zone_states[i, 2] == pytest.approx(0.5 * 250.0)

    def test_routing_buffer_shape(self, valid_params: Parameters) -> None:
        """Routing buffer has correct shape."""
        state = State.initialize(valid_params)
        assert state.routing_buffer.shape == (7,)
        np.testing.assert_array_equal(state.routing_buffer, np.zeros(7))

    def test_array_roundtrip(self, valid_params: Parameters) -> None:
        """State can be reconstructed from array."""
        original = State.initialize(valid_params)
        original.zone_states[0, 0] = 50.0  # Set snow pack
        original.zone_states[0, 2] = 200.0  # Set soil moisture
        original.upper_zone = 10.0
        original.lower_zone = 5.0
        original.routing_buffer[0] = 2.0

        arr = np.asarray(original)
        restored = State.from_array(arr, n_zones=1)

        np.testing.assert_array_almost_equal(restored.zone_states, original.zone_states)
        assert restored.upper_zone == original.upper_zone
        assert restored.lower_zone == original.lower_zone
        np.testing.assert_array_almost_equal(restored.routing_buffer, original.routing_buffer)

    def test_array_roundtrip_multi_zone(self, valid_params: Parameters) -> None:
        """State array roundtrip works with multiple zones."""
        original = State.initialize(valid_params, n_zones=2)
        original.zone_states[0, 0] = 30.0
        original.zone_states[1, 0] = 50.0
        original.upper_zone = 15.0

        arr = np.asarray(original)
        restored = State.from_array(arr, n_zones=2)

        np.testing.assert_array_almost_equal(restored.zone_states, original.zone_states)
        assert restored.upper_zone == original.upper_zone
