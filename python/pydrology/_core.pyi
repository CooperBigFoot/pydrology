import numpy

def rust_version() -> str: ...

class cemaneige:
    @staticmethod
    def gr6j_cemaneige_run(
        params: numpy.ndarray,
        precip: numpy.ndarray,
        pet: numpy.ndarray,
        temp: numpy.ndarray,
        initial_state: numpy.ndarray | None = None,
        uh1_ordinates: numpy.ndarray | None = None,
        uh2_ordinates: numpy.ndarray | None = None,
        n_layers: int = 1,
        layer_elevations: numpy.ndarray | None = None,
        layer_fractions: numpy.ndarray | None = None,
        input_elevation: float | None = None,
        temp_gradient: float | None = None,
        precip_gradient: float | None = None,
        mean_annual_solid_precip: float = 0.0,
    ) -> dict[str, numpy.ndarray]: ...
    @staticmethod
    def gr6j_cemaneige_step(
        state: numpy.ndarray,
        params: numpy.ndarray,
        precip: float,
        pet: float,
        temp: float,
        uh1_ordinates: numpy.ndarray,
        uh2_ordinates: numpy.ndarray,
        layer_elevations: numpy.ndarray,
        layer_fractions: numpy.ndarray,
        input_elevation: float | None = None,
        temp_gradient: float | None = None,
        precip_gradient: float | None = None,
    ) -> tuple[numpy.ndarray, dict[str, float]]: ...

class gr2m:
    @staticmethod
    def gr2m_run(
        params: numpy.ndarray,
        precip: numpy.ndarray,
        pet: numpy.ndarray,
        initial_state: numpy.ndarray | None = None,
    ) -> dict[str, numpy.ndarray]: ...
    @staticmethod
    def gr2m_step(
        state: numpy.ndarray,
        params: numpy.ndarray,
        precip: float,
        pet: float,
    ) -> tuple[numpy.ndarray, dict[str, float]]: ...

class hbv_light:
    @staticmethod
    def hbv_run(
        params: numpy.ndarray,
        precip: numpy.ndarray,
        pet: numpy.ndarray,
        temp: numpy.ndarray,
        initial_state: numpy.ndarray | None = None,
        n_zones: int = 1,
        zone_elevations: numpy.ndarray | None = None,
        zone_fractions: numpy.ndarray | None = None,
        input_elevation: float | None = None,
        temp_gradient: float | None = None,
        precip_gradient: float | None = None,
    ) -> tuple[dict[str, numpy.ndarray], dict[str, object] | None]: ...
    @staticmethod
    def hbv_step(
        state: numpy.ndarray,
        params: numpy.ndarray,
        precip: float,
        pet: float,
        temp: float,
        uh_weights: numpy.ndarray,
    ) -> tuple[numpy.ndarray, dict[str, float]]: ...
    @staticmethod
    def hbv_triangular_weights(
        maxbas: float,
    ) -> numpy.ndarray: ...
