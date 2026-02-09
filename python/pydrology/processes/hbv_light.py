"""HBV-light process functions.

Combines snow/soil moisture processes and response/routing functions.
"""

import math

import numpy as np

# === Snow and soil moisture processes ===


def partition_precipitation(precip: float, temp: float, tt: float, sfcf: float) -> tuple[float, float]:
    """Partition precipitation into rain and snow.

    Args:
        precip: Total precipitation [mm/day].
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        sfcf: Snowfall correction factor [-].

    Returns:
        Tuple of (p_rain, p_snow) in mm/day.
    """
    if temp > tt:
        return precip, 0.0
    else:
        return 0.0, sfcf * precip


def compute_melt(temp: float, tt: float, cfmax: float, snow_pack: float) -> float:
    """Compute snowmelt using degree-day method.

    Args:
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        cfmax: Degree-day factor [mm/deg C/day].
        snow_pack: Current snow pack [mm].

    Returns:
        Melt rate [mm/day], limited by available snow.
    """
    if temp > tt:
        melt = cfmax * (temp - tt)
        return min(melt, snow_pack)
    return 0.0


def compute_refreezing(temp: float, tt: float, cfmax: float, cfr: float, liquid_water: float) -> float:
    """Compute refreezing of liquid water in snowpack.

    Args:
        temp: Air temperature [deg C].
        tt: Threshold temperature [deg C].
        cfmax: Degree-day factor [mm/deg C/day].
        cfr: Refreezing coefficient [-].
        liquid_water: Liquid water in snowpack [mm].

    Returns:
        Refreezing rate [mm/day], limited by available liquid water.
    """
    if temp < tt:
        refreeze = cfr * cfmax * (tt - temp)
        return min(refreeze, liquid_water)
    return 0.0


def update_snow_pack(
    sp: float,
    lw: float,
    p_snow: float,
    melt: float,
    refreeze: float,
    cwh: float,
) -> tuple[float, float, float]:
    """Update snow pack and liquid water states.

    Args:
        sp: Current snow pack [mm].
        lw: Current liquid water in snow [mm].
        p_snow: Snowfall [mm/day].
        melt: Snowmelt [mm/day].
        refreeze: Refreezing [mm/day].
        cwh: Water holding capacity of snow [-].

    Returns:
        Tuple of (new_sp, new_lw, outflow):
        - new_sp: Updated snow pack [mm]
        - new_lw: Updated liquid water in snow [mm]
        - outflow: Water released from snow to soil [mm/day]
    """
    # Update snow pack: add snowfall, subtract melt, add refrozen water
    new_sp = sp + p_snow - melt + refreeze

    # Update liquid water: add melt, subtract refreezing
    new_lw = lw + melt - refreeze

    # Compute maximum liquid water storage
    lw_max = cwh * new_sp

    # Compute outflow (excess liquid water)
    if new_lw > lw_max:
        outflow = new_lw - lw_max
        new_lw = lw_max
    else:
        outflow = 0.0

    # Ensure non-negative
    new_sp = max(0.0, new_sp)
    new_lw = max(0.0, new_lw)

    return new_sp, new_lw, outflow


def compute_recharge(soil_input: float, sm: float, fc: float, beta: float) -> float:
    """Compute groundwater recharge from soil.

    Uses the HBV soil moisture / runoff relationship:
    Recharge/Input = (SM/FC)^BETA

    Args:
        soil_input: Water input to soil (rain + snowmelt) [mm/day].
        sm: Current soil moisture [mm].
        fc: Field capacity [mm].
        beta: Shape coefficient [-].

    Returns:
        Recharge to groundwater [mm/day].
    """
    if fc <= 0.0 or soil_input <= 0.0:
        return 0.0

    # Saturation ratio, clamped to [0, 1]
    sr = min(max(sm / fc, 0.0), 1.0)

    # Recharge fraction
    recharge = soil_input * (sr**beta)

    return recharge


def compute_actual_et(pet: float, sm: float, fc: float, lp: float) -> float:
    """Compute actual evapotranspiration from soil.

    ET is reduced linearly when SM < LP * FC.

    Args:
        pet: Potential evapotranspiration [mm/day].
        sm: Current soil moisture [mm].
        fc: Field capacity [mm].
        lp: Limit for potential ET as fraction of FC [-].

    Returns:
        Actual evapotranspiration [mm/day].
    """
    if fc <= 0.0 or lp <= 0.0:
        return 0.0

    lp_threshold = lp * fc

    # Linear reduction below threshold, full PET above
    et_act = pet if sm >= lp_threshold else pet * sm / lp_threshold

    # Cannot extract more than available
    et_act = min(et_act, max(sm, 0.0))

    return et_act


def update_soil_moisture(
    sm: float,
    soil_input: float,
    recharge: float,
    et_act: float,
    fc: float,
) -> float:
    """Update soil moisture state.

    Args:
        sm: Current soil moisture [mm].
        soil_input: Water input to soil [mm/day].
        recharge: Recharge leaving to groundwater [mm/day].
        et_act: Actual evapotranspiration [mm/day].
        fc: Field capacity [mm].

    Returns:
        Updated soil moisture [mm], bounded [0, FC].
    """
    # Infiltration is the non-recharge portion
    infiltration = soil_input - recharge

    # Update soil moisture
    new_sm = sm + infiltration - et_act

    # Bound to [0, FC]
    new_sm = max(0.0, min(new_sm, fc))

    return new_sm


# === Response and routing functions ===


def upper_zone_outflows(suz: float, k0: float, k1: float, uzl: float) -> tuple[float, float]:
    """Compute outflows from upper groundwater zone.

    Q0 (surface flow) is threshold-activated when SUZ > UZL.
    Q1 (interflow) is always active.

    Args:
        suz: Upper zone storage [mm].
        k0: Surface flow recession coefficient [1/day].
        k1: Interflow recession coefficient [1/day].
        uzl: Upper zone threshold for Q0 [mm].

    Returns:
        Tuple of (q0, q1) in mm/day.
    """
    # Q0: threshold-activated surface flow
    q0 = k0 * (suz - uzl) if suz > uzl else 0.0

    # Q1: always-active interflow
    q1 = k1 * suz

    return q0, q1


def compute_percolation(suz: float, perc_max: float) -> float:
    """Compute percolation from upper to lower zone.

    Args:
        suz: Upper zone storage [mm].
        perc_max: Maximum percolation rate [mm/day].

    Returns:
        Actual percolation [mm/day], limited by available storage.
    """
    return min(perc_max, max(suz, 0.0))


def update_upper_zone(suz: float, recharge: float, q0: float, q1: float, perc: float) -> float:
    """Update upper zone storage.

    Args:
        suz: Current upper zone storage [mm].
        recharge: Inflow from soil [mm/day].
        q0: Surface flow outflow [mm/day].
        q1: Interflow outflow [mm/day].
        perc: Percolation to lower zone [mm/day].

    Returns:
        Updated upper zone storage [mm], non-negative.
    """
    new_suz = suz + recharge - q0 - q1 - perc
    return max(0.0, new_suz)


def lower_zone_outflow(slz: float, k2: float) -> float:
    """Compute baseflow from lower groundwater zone.

    Args:
        slz: Lower zone storage [mm].
        k2: Baseflow recession coefficient [1/day].

    Returns:
        Baseflow Q2 [mm/day].
    """
    return k2 * slz


def update_lower_zone(slz: float, perc: float, q2: float) -> float:
    """Update lower zone storage.

    Args:
        slz: Current lower zone storage [mm].
        perc: Percolation inflow from upper zone [mm/day].
        q2: Baseflow outflow [mm/day].

    Returns:
        Updated lower zone storage [mm], non-negative.
    """
    new_slz = slz + perc - q2
    return max(0.0, new_slz)


def compute_triangular_weights(maxbas: float) -> np.ndarray:
    """Compute triangular unit hydrograph weights.

    Uses analytical integration of the triangular function to compute
    weights that sum to 1.0. Supports fractional MAXBAS values.

    The triangular function has:
    - Peak at t = MAXBAS/2
    - Rising limb from t=0 to t=MAXBAS/2
    - Falling limb from t=MAXBAS/2 to t=MAXBAS

    Args:
        maxbas: Length of triangular function [days].

    Returns:
        Array of weights with length ceil(MAXBAS), summing to 1.0.
    """
    n = int(math.ceil(maxbas))
    if n < 1:
        n = 1

    weights = np.zeros(n, dtype=np.float64)
    half = maxbas / 2.0

    for i in range(n):
        t_start = float(i)
        t_end = min(float(i + 1), maxbas)

        if t_end <= t_start:
            continue

        # Integrate the triangular function over [t_start, t_end]
        # Rising limb: f(t) = 2t / MAXBAS^2, for t < MAXBAS/2
        # Falling limb: f(t) = 2/MAXBAS - 2t / MAXBAS^2, for t >= MAXBAS/2

        weight = 0.0

        # Handle rising limb portion
        if t_start < half:
            t_r_end = min(t_end, half)
            # Integral of 2t/MAXBAS^2 from t_start to t_r_end
            # = (t_r_end^2 - t_start^2) / MAXBAS^2
            weight += (t_r_end**2 - t_start**2) / (maxbas**2)

        # Handle falling limb portion
        if t_end > half:
            t_f_start = max(t_start, half)
            # Integral of (2/MAXBAS - 2t/MAXBAS^2) from t_f_start to t_end
            # = 2*(t_end - t_f_start)/MAXBAS - (t_end^2 - t_f_start^2)/MAXBAS^2
            weight += 2.0 * (t_end - t_f_start) / maxbas - (t_end**2 - t_f_start**2) / (maxbas**2)

        weights[i] = weight

    # Normalize to ensure sum = 1.0 (handles numerical precision)
    total = weights.sum()
    if total > 0:
        weights /= total

    return weights


def convolve_triangular(qgw: float, buffer: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Convolve groundwater runoff with triangular unit hydrograph.

    Uses a buffer to store delayed contributions. The first element
    of the buffer becomes the output, then the buffer is shifted.

    Args:
        qgw: Current groundwater runoff [mm/day].
        buffer: Routing buffer, length 7 (MAXBAS_MAX).
        weights: Triangular UH weights from compute_triangular_weights().

    Returns:
        Tuple of (new_buffer, qsim):
        - new_buffer: Updated routing buffer
        - qsim: Routed streamflow [mm/day]
    """
    # Output is first buffer element
    qsim = buffer[0]

    # Create new buffer, shifted left
    n_buf = len(buffer)
    new_buffer = np.zeros(n_buf, dtype=np.float64)

    # Shift buffer left
    for i in range(n_buf - 1):
        new_buffer[i] = buffer[i + 1]

    # Add current input contribution using weights
    n_weights = len(weights)
    for i in range(min(n_weights, n_buf)):
        new_buffer[i] += qgw * weights[i]

    return new_buffer, qsim
