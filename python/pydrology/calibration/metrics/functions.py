"""Hydrological metrics for calibration objectives.

All metrics take (observed, simulated) arrays and return a scalar score.
Backed by Rust implementations via PyO3.
"""

import numpy as np
from numpy.typing import ArrayLike

from pydrology._core.metrics import (
    rust_kge,
    rust_log_nse,
    rust_mae,
    rust_nse,
    rust_pbias,
    rust_rmse,
)

from .registry import register


@register("maximize")
def nse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency.

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)

    Range: (-inf, 1], where 1 is perfect match.
    """
    return float(rust_nse(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))


@register("maximize")
def log_nse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Nash-Sutcliffe Efficiency on log-transformed flows.

    Emphasizes low-flow performance by log-transforming both series.
    Small constant (1e-6) added to avoid log(0).

    Range: (-inf, 1], where 1 is perfect match.
    """
    return float(rust_log_nse(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))


@register("maximize")
def kge(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Kling-Gupta Efficiency.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)

    Where:
        r = Pearson correlation coefficient
        alpha = std(sim) / std(obs)
        beta = mean(sim) / mean(obs)

    Range: (-inf, 1], where 1 is perfect match.
    """
    return float(rust_kge(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))


@register("minimize")
def pbias(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Percent Bias.

    PBIAS = 100 * sum(sim - obs) / sum(obs)

    Positive PBIAS = overestimation, negative = underestimation.
    Optimal value is 0.
    """
    return float(rust_pbias(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))


@register("minimize")
def rmse(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Root Mean Square Error.

    RMSE = sqrt(mean((obs - sim)^2))

    Range: [0, inf), where 0 is perfect match.
    """
    return float(rust_rmse(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))


@register("minimize")
def mae(observed: ArrayLike, simulated: ArrayLike) -> float:
    """Mean Absolute Error.

    MAE = mean(|obs - sim|)

    Range: [0, inf), where 0 is perfect match.
    """
    return float(rust_mae(np.ascontiguousarray(observed, dtype=np.float64), np.ascontiguousarray(simulated, dtype=np.float64)))
