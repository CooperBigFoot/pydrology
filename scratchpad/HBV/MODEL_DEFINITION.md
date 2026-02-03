# HBV Model Technical Definition

> **Hydrologiska Byråns Vattenbalansavdelning**
> A lumped (or semi-distributed) conceptual rainfall-runoff model for daily streamflow simulation

This document describes the HBV model structure, equations, and implementation details compiled from multiple sources including the Seibert & Vis (2012) paper and four open-source implementations.

---

## Table of Contents

1. [Model Overview](#1-model-overview)
   - [History and Development](#history-and-development)
   - [Model Variants](#model-variants)
   - [Warm-up Period](#warm-up-period)
2. [Model Parameters](#2-model-parameters)
3. [State Variables](#3-state-variables)
4. [Model Structure Diagram](#4-model-structure-diagram)
5. [Mathematical Equations](#5-mathematical-equations)
   - [5.0 Input Distribution (Semi-Distributed Mode)](#50-input-distribution-semi-distributed-mode---optional)
   - [5.1 Snow Routine](#51-snow-routine)
   - [5.2 Soil Moisture Routine](#52-soil-moisture-routine)
   - [5.3 Response Routine](#53-response-routine)
   - [5.4 Routing Routine](#54-routing-routine)
6. [Complete Algorithm](#6-complete-algorithm)
7. [Numerical Constants](#7-numerical-constants)
8. [Model Outputs](#8-model-outputs)
9. [Implementation Variants](#9-implementation-variants)
10. [References](#10-references)
11. [Appendix A: Parameter Bounds Summary](#appendix-a-parameter-bounds-summary)
    - [Calibration Notes: Parameter Interactions](#calibration-notes-parameter-interactions-equifinality)
12. [Appendix B: Symbol Reference](#appendix-b-symbol-reference)

---

## 1. Model Overview

**HBV** (Hydrologiska Byråns Vattenbalansavdelning, meaning "Water Balance Section of the Hydrological Bureau") is a widely-used conceptual rainfall-runoff model developed for catchment hydrology simulations. It simulates discharge based on precipitation, air temperature, and potential evapotranspiration.

### Key Characteristics

| Property | Value |
|----------|-------|
| Time step | Daily (primary), can be sub-daily |
| Spatial resolution | Lumped (or semi-distributed with elevation zones) |
| Parameters | 9-15 calibrated parameters (variant-dependent) |
| Stores | 4 (Snow, Soil, Upper Zone, Lower Zone) |
| Unit hydrograph | Triangular weighting function |
| Inputs | Precipitation (P), Temperature (T), Potential ET (E) |
| Output | Streamflow at catchment outlet (Q) |

### History and Development

- **1970s**: Original development at the Swedish Meteorological and Hydrological Institute (SMHI) by Sten Bergström
- **1976**: First major publication (Bergström, 1976)
- **1992**: HBV-ETH variant developed at ETH Zurich (Braun & Renner)
- **1993**: HBV-light developed at Uppsala University by Seibert
- **1997**: HBV-96 distributed version (Lindström et al.)
- **2012**: Enhanced HBV-light at University of Zurich (Seibert & Vis)

### Model Variants

| Variant | Description | Key Feature |
|---------|-------------|-------------|
| **HBV-96** | Distributed version by SMHI | Spatial discretization |
| **HBV-light** | Educational version | User-friendly GUI |
| **HBV-ETH** | Swiss variant | Glacier routines |
| **HBV-EDU** | Teaching version | Simplified structure |

### Model Philosophy

HBV represents a **deliberate compromise between complexity and interpretability**:

- Conceptual (not black-box empirical, not fully physically-based)
- Transparent processes that are readily understandable
- Moderate data requirements
- Computationally efficient

### Warm-up Period

| Aspect | Recommendation |
|--------|----------------|
| Duration | **365 days** (1 year) minimum |
| Extended | **2-3 years** for catchments with large groundwater reservoirs (low K2) or deep seasonal snowpacks |
| Purpose | Allow stores to reach dynamic equilibrium |
| Outputs | Should be discarded from analysis |

**Why warm-up is needed:**

- Initial store levels (SM, SUZ, SLZ, Snow) are set to default values
- These defaults rarely match actual catchment conditions
- The warm-up allows model states to "spin up" to realistic values
- The lower zone (SLZ) with slow recession (K2 ~ 0.01/d) may require 2+ years to stabilize

---

## 2. Model Parameters

HBV uses **9-15 calibrated parameters** depending on the variant. The core parameter set includes:

### Snow Routine Parameters

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **TT** | Threshold temperature (rain/snow) | °C | [-1.5, 2.5] |
| **CFMAX** | Degree-day factor (melt rate) | mm/°C/d | [1, 10] |
| **CFR** | Refreezing coefficient | - | [0, 0.1] |
| **CWH** | Water holding capacity of snow | - | [0, 0.2] |
| **SFCF** | Snowfall correction factor | - | [0.4, 1.0] |

### Soil Routine Parameters

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **FC** | Maximum soil moisture storage | mm | [50, 500] |
| **LP** | Evaporation reduction threshold | - | [0.3, 1.0] |
| **BETA** | Shape coefficient | - | [1, 6] |

### Response Routine Parameters

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **K0** | Recession coefficient (surface) | 1/d | [0.1, 0.5] |
| **K1** | Recession coefficient (upper zone) | 1/d | [0.01, 0.4] |
| **K2** | Recession coefficient (lower zone) | 1/d | [0.001, 0.15] |
| **UZL** | Threshold for K0 activation | mm | [0, 100] |
| **PERC** | Maximum percolation rate | mm/d | [0, 6] |

### Routing Parameter

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| **MAXBAS** | Length of triangular weighting function | d | [1, 7] |

### Physical Interpretation

- **TT (Threshold temperature)**: Determines whether precipitation falls as rain or snow. Values typically near 0°C but can vary with local conditions.

- **CFMAX (Degree-day factor)**: Controls how fast snow melts per degree above threshold. Higher values = faster melt. Varies with vegetation, aspect, and elevation.

- **FC (Field capacity)**: Maximum water the soil can hold against gravity. Controls the total "bucket size" for soil moisture accounting.

- **BETA (Shape coefficient)**: Controls the non-linearity of runoff generation. Higher values mean more water must accumulate before significant runoff occurs.

- **LP (Limit for potential ET)**: Fraction of FC above which actual ET equals potential ET. Below this threshold, ET is reduced linearly.

- **K0, K1, K2 (Recession coefficients)**: Control how quickly water drains from each reservoir. K0 > K1 > K2 typically, representing fast to slow flow paths.

- **UZL (Upper zone threshold)**: Water level in upper zone that must be exceeded before K0 (fastest) flow activates.

- **PERC (Percolation)**: Maximum daily transfer from upper to lower groundwater zone.

- **MAXBAS (Routing time)**: Length of the triangular transfer function for channel routing. Larger values = more flow attenuation.

---

## 3. State Variables

The model maintains **4 primary storage compartments**:

| Store | Symbol | Description | Initialization |
|-------|--------|-------------|----------------|
| Snow pack | SP / SWE | Snow water equivalent | 0 mm |
| Soil moisture | SM | Water in root zone | 0.5 × FC |
| Upper zone | SUZ / S1 | Fast response storage | 0 mm |
| Lower zone | SLZ / S2 | Slow response storage | 0 mm |

### Additional State Variables (Some Variants)

| Variable | Description |
|----------|-------------|
| Liquid water in snow | Water held in snowpack |
| Interception storage (Si) | Water on vegetation (HBV-light, HBV-bmi) |

> **Note on Interception:** Most simplified implementations omit the interception store. When present (e.g., HBV-bmi), evaporation is first taken from interception storage, then the remaining demand is applied to soil moisture as transpiration.

---

## 4. Model Structure Diagram

```
                         INPUTS
                           │
              ┌────────────┴────────────┐
              │                         │
        Precipitation (P)         Temperature (T)
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │      SNOW ROUTINE      │
              │                        │
              │  • Rain/snow partition │
              │  • Snow accumulation   │
              │  • Degree-day melt     │
              │  • Refreezing          │
              │  [Parameters: TT,      │
              │   CFMAX, CFR, CWH]     │
              └────────────┬───────────┘
                           │
                    Rainfall + Snowmelt
                           │
                           ▼
              ┌────────────────────────┐
              │     SOIL ROUTINE       │
              │    (capacity: FC)      │
              │                        │
              │  • Infiltration        │
              │  • Evapotranspiration  │
              │  • Recharge to GW      │
              │  [Parameters: FC,      │
              │   BETA, LP]            │
              └────────────┬───────────┘
                           │
                      Recharge
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         │
    ┌─────────────────┐                 │
    │   UPPER ZONE    │                 │
    │     (SUZ)       │                 │
    │                 │                 │
    │ Q0 = K0×(SUZ-UZL)│◄── if SUZ>UZL │
    │ Q1 = K1 × SUZ   │                 │
    │ [K0, K1, UZL]   │                 │
    └────────┬────────┘                 │
             │                          │
        Percolation                     │
         (PERC)                         │
             │                          │
             ▼                          │
    ┌─────────────────┐                 │
    │   LOWER ZONE    │                 │
    │     (SLZ)       │                 │
    │                 │                 │
    │ Q2 = K2 × SLZ   │                 │
    │ [K2]            │                 │
    └────────┬────────┘                 │
             │                          │
             ▼                          │
    ┌─────────────────┐                 │
    │  TOTAL RUNOFF   │◄────────────────┘
    │                 │
    │ QGW = Q0+Q1+Q2  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ ROUTING ROUTINE │
    │                 │
    │ Triangular UH   │
    │ [MAXBAS]        │
    └────────┬────────┘
             │
             ▼
          OUTPUT
        Qsim (m³/s)
```

---

## 5. Mathematical Equations

### 5.0 Input Distribution (Semi-Distributed Mode - Optional)

When HBV is run in **semi-distributed mode**, the catchment is divided into elevation zones. The snow and soil routines are computed separately for each zone, while the response routine operates on the aggregated recharge.

#### Temperature Lapse Rate

Temperature is adjusted for each elevation zone:

$$T_{zone} = T_{station} - TCALT \cdot \frac{\Delta h}{100}$$

Where:
- T_zone = adjusted temperature for the zone (°C)
- T_station = measured temperature at the climate station (°C)
- TCALT = temperature lapse rate (typically **0.6 °C per 100m**)
- Δh = elevation difference between zone and station (m)

#### Precipitation Correction

Precipitation is adjusted for orographic effects:

$$P_{zone} = P_{station} \cdot \left(1 + PCALT \cdot \frac{\Delta h}{100}\right)$$

Where:
- P_zone = adjusted precipitation for the zone (mm/d)
- P_station = measured precipitation at the climate station (mm/d)
- PCALT = precipitation correction factor (typically **10-20% per 100m**)
- Δh = elevation difference between zone and station (m)

#### Zone Aggregation

After computing snow and soil routines per zone, recharge is aggregated by area-weighting:

$$Recharge_{total} = \sum_{z=1}^{n} Recharge_z \cdot \frac{Area_z}{Area_{total}}$$

> **Implementation Note:** Most modern Python implementations (lumod, HBV-bmi, hbv_hydromodel) operate in **lumped mode only** and do not include elevation zone calculations. The semi-distributed equations above are found primarily in the original SMHI implementations and the C-based HBV code.

---

### 5.1 Snow Routine

The snow routine uses the **degree-day method** for snow accumulation and melt.

#### Precipitation Partitioning

$$P_{rain} = \begin{cases} P & \text{if } T > TT \\ 0 & \text{if } T \leq TT \end{cases}$$

$$P_{snow} = \begin{cases} 0 & \text{if } T > TT \\ SFCF \cdot P & \text{if } T \leq TT \end{cases}$$

Where:

- P = precipitation (mm/d)
- T = air temperature (°C)
- TT = threshold temperature (°C)
- SFCF = snowfall correction factor (accounts for undercatch)

#### Snow Accumulation

$$SP_{new} = SP + P_{snow}$$

#### Snowmelt (Degree-Day Method)

$$M = CFMAX \cdot (T - TT) \quad \text{when } T > TT$$

$$M = \min(M, SP) \quad \text{(cannot melt more than exists)}$$

Where:

- M = melt rate (mm/d)
- CFMAX = degree-day factor (mm/°C/d)

#### Refreezing

When temperature is below threshold, liquid water in snowpack can refreeze:

$$R = CFR \cdot CFMAX \cdot (TT - T) \quad \text{when } T < TT$$

Where:

- R = refreezing rate (mm/d)
- CFR = refreezing coefficient (typically 0.05, making refreezing ~20× slower than melt)

> **Implementation Note:** Many simplified implementations (lumod, HBV-bmi, hbv_hydromodel) **omit refreezing entirely**, using only the degree-day melt equation. The full refreezing formulation is found in HBV-light and the original SMHI code.

#### Water Holding Capacity

Snowpack can retain liquid water up to a fraction of its water equivalent:

$$LW_{max} = CWH \cdot SP$$

Where:

- LW_max = maximum liquid water storage in snow (mm)
- CWH = water holding capacity (typically 0.1)

#### Effective Input to Soil

$$I = P_{rain} + M - R + \max(LW - LW_{max}, 0)$$

---

### 5.2 Soil Moisture Routine

The soil routine partitions water between infiltration, evapotranspiration, and recharge.

#### Groundwater Recharge (Runoff Generation)

The recharge to groundwater is a non-linear function of soil moisture:

$$\frac{Recharge}{I} = \left(\frac{SM}{FC}\right)^{BETA}$$

Rearranged:
$$Recharge = I \cdot \left(\frac{SM}{FC}\right)^{BETA}$$

Where:

- I = input (rainfall + snowmelt) (mm/d)
- SM = current soil moisture (mm)
- FC = field capacity (mm)
- BETA = shape coefficient (-)

**Physical interpretation:**

- When SM is low (dry soil): most water infiltrates, little recharge
- When SM approaches FC (wet soil): most water becomes recharge
- BETA controls the curvature of this relationship

#### Infiltration

Water not becoming recharge is added to soil storage:

$$Infiltration = I - Recharge = I \cdot \left(1 - \left(\frac{SM}{FC}\right)^{BETA}\right)$$

#### Soil Moisture Update

$$SM_{new} = SM + Infiltration - E_{act}$$

#### Actual Evapotranspiration

Actual ET depends on soil moisture availability:

$$E_{act} = \begin{cases} E_{pot} & \text{if } SM \geq LP \cdot FC \\ E_{pot} \cdot \frac{SM}{LP \cdot FC} & \text{if } SM < LP \cdot FC \end{cases}$$

Where:

- E_pot = potential evapotranspiration (mm/d)
- LP = limit for potential ET (fraction of FC)

**Physical interpretation:**

- Above the LP threshold: soil is wet enough for unrestricted ET
- Below the LP threshold: ET reduces linearly as soil dries

#### Potential Evapotranspiration Correction (Optional)

Some variants adjust PE based on temperature deviation:

$$E_{pot}(t) = \left(1 + C_{ET} \cdot (T(t) - T_M)\right) \cdot E_{pot,M}$$

With bounds:
$$0 \leq E_{pot}(t) \leq 2 \cdot E_{pot,M}$$

Where:

- C_ET = temperature correction factor (°C⁻¹)
- T_M = long-term mean temperature for day of year
- E_pot,M = long-term mean PE for day of year

---

### 5.3 Response Routine

The response routine converts recharge to runoff using two linear reservoirs.

#### Upper Zone (SUZ) Dynamics

**Inflow:**
$$SUZ_{new} = SUZ + Recharge$$

**Outflows (three components):**

1. **Surface/Quick flow (Q0)** - threshold-activated:
$$Q_0 = K_0 \cdot \max(SUZ - UZL, 0)$$

2. **Interflow (Q1)** - always active:
$$Q_1 = K_1 \cdot SUZ$$

3. **Percolation to lower zone:**
$$PERC_{actual} = \min(PERC, SUZ)$$

**Upper zone update:**
$$SUZ_{final} = SUZ - Q_0 - Q_1 - PERC_{actual}$$

#### Lower Zone (SLZ) Dynamics

**Inflow:**
$$SLZ_{new} = SLZ + PERC_{actual}$$

**Outflow (baseflow):**
$$Q_2 = K_2 \cdot SLZ$$

**Lower zone update:**
$$SLZ_{final} = SLZ - Q_2$$

#### Total Groundwater Runoff

$$Q_{GW} = Q_0 + Q_1 + Q_2$$

---

### 5.4 Routing Routine

The routing routine smooths the runoff response using a **triangular unit hydrograph**.

#### Triangular Weighting Function

The weights c(i) form a symmetric triangle over MAXBAS days:

$$c(i) = \int_{i-1}^{i} \left[\frac{2}{MAXBAS} - \left|u - \frac{MAXBAS}{2}\right| \cdot \frac{4}{MAXBAS^2}\right] du$$

Simplified discrete form:

- Peak at t = MAXBAS/2
- Rising limb: c(i) increases linearly from 0 to peak
- Falling limb: c(i) decreases linearly from peak to 0
- Sum of all weights = 1.0

#### Convolution

$$Q_{sim}(t) = \sum_{i=1}^{MAXBAS} c(i) \cdot Q_{GW}(t - i + 1)$$

> **Implementation Note (Non-Integer MAXBAS):** When MAXBAS is not an integer (e.g., 2.5 days), implementations vary:
> - **lumod**: Uses `ceil(MAXBAS)` for array size → MAXBAS=2.5 gives 3 weights
> - **HBV-bmi**: Rounds to nearest integer → MAXBAS=2.5 gives 3, but MAXBAS=2.4 gives 2
>
> This is a common source of discrepancies between implementations. For reproducibility, consider using integer MAXBAS values or documenting the specific rounding behavior.

#### Unit Conversion

Convert from mm/d to m³/s:

$$Q_{m^3/s} = Q_{mm/d} \cdot \frac{Area_{km^2} \cdot 1000}{86400}$$

Simplified:
$$Q_{m^3/s} = Q_{mm/d} \cdot \frac{Area}{86.4}$$

---

## 6. Complete Algorithm

```
INPUT: P (precipitation), T (temperature), E_pot (potential ET)
STATE: SP (snow), SM (soil), SUZ (upper zone), SLZ (lower zone)

FOR each time step:

  1. SNOW ROUTINE
     IF T < TT:
       P_snow = SFCF * P
       P_rain = 0
       SP = SP + P_snow
       Refreezing (if liquid water present)
     ELSE:
       P_rain = P
       P_snow = 0
       Melt = min(SP, CFMAX * (T - TT))
       SP = SP - Melt
     ENDIF

     Input = P_rain + Melt (after liquid water accounting)

  2. SOIL ROUTINE
     SR = SM / FC
     Recharge = Input * SR^BETA
     Infiltration = Input - Recharge
     SM = SM + Infiltration

     IF SM > LP * FC:
       E_act = E_pot
     ELSE:
       E_act = E_pot * SM / (LP * FC)
     ENDIF

     SM = max(0, SM - E_act)

  3. RESPONSE ROUTINE
     SUZ = SUZ + Recharge

     Q0 = K0 * max(0, SUZ - UZL)
     Q1 = K1 * SUZ
     Percolation = min(PERC, SUZ)

     SUZ = SUZ - Q0 - Q1 - Percolation

     SLZ = SLZ + Percolation
     Q2 = K2 * SLZ
     SLZ = SLZ - Q2

     Q_GW = Q0 + Q1 + Q2

  4. ROUTING ROUTINE
     Apply triangular weighting function with MAXBAS
     Q_sim = convolve(Q_GW, triangular_weights)

  5. UNIT CONVERSION
     Q_final = Q_sim * Area / 86.4

OUTPUT: Q_final (streamflow in m³/s)
```

---

## 7. Numerical Constants

| Constant | Value | Description |
|----------|-------|-------------|
| Default TT | 0.0 | Threshold temperature (°C) |
| Default CFR | 0.05 | Refreezing ~20× slower than melt |
| Default CWH | 0.1 | 10% water holding in snow |
| Default SFCF | 1.0 | No snowfall correction |
| Default LP | 0.9 | ET threshold at 90% of FC |
| Default BETA | 2.0 | Quadratic soil response |
| Max MAXBAS | 7-10 | Upper limit for routing |
| Min MAXBAS | 1 | At least 1 day routing |

---

## 8. Model Outputs

| Output | Symbol | Description | Unit |
|--------|--------|-------------|------|
| Qsim | Q | Total simulated streamflow | m³/s or mm/d |
| Snow | SP | Snow water equivalent | mm |
| Soil moisture | SM | Soil moisture storage | mm |
| Upper zone | SUZ | Upper groundwater storage | mm |
| Lower zone | SLZ | Lower groundwater storage | mm |
| Actual ET | E_act | Actual evapotranspiration | mm/d |
| Snowmelt | M | Daily snowmelt | mm/d |
| Q0 | Q_0 | Surface/quick flow | mm/d |
| Q1 | Q_1 | Interflow | mm/d |
| Q2 | Q_2 | Baseflow | mm/d |

---

## 9. Implementation Variants

### Standard HBV (9-10 parameters)

The minimal implementation with:

- Snow: TT, CFMAX
- Soil: FC, BETA, LP (or PWP)
- Response: K0, K1, K2, UZL, PERC
- Routing: MAXBAS

### HBV-light (15 parameters)

Adds:

- Interception storage (Imax)
- Multiple PET calculation methods
- Glacier routine
- Extended snow parameters (CFR, CWH, SFCF)

### Simplified HBV (Educational)

Some implementations reduce to 8-9 parameters by:

- Removing K0 (no threshold flow)
- Combining refreezing into CFMAX
- Using fixed LP = 1.0

### Response Routine Variants

| Variant | Description |
|---------|-------------|
| **Two-box linear** | Standard SUZ + SLZ with linear outflow |
| **One-box** | Single reservoir (SUZ only) |
| **Three-box** | Additional storage for deep groundwater |
| **Non-linear** | Power function: Q = K × S^(1+α) |
| **Split recharge** | Divides recharge between fast/slow paths |

---

## 10. References

### Primary References

> Bergström, S. (1976). **Development and application of a conceptual runoff model for Scandinavian catchments.** SMHI Reports RHO, No. 7, Norrköping.

> Bergström, S. (1995). **The HBV model.** In: Singh, V.P. (Ed.), Computer Models of Watershed Hydrology. Water Resources Publications, Highlands Ranch, CO, pp. 443-476.

> Seibert, J. & Vis, M.J.P. (2012). **Teaching hydrological modeling with a user-friendly catchment-runoff-model software package.** Hydrology and Earth System Sciences, 16, 3315-3325.

### Related Publications

- Lindström, G., Johansson, B., Persson, M., Gardelin, M., & Bergström, S. (1997). Development and test of the distributed HBV-96 hydrological model. *Journal of Hydrology*, 201, 272-288.

- Braun, L.N. & Renner, C.B. (1992). Application of a conceptual runoff model in different physiographic regions of Switzerland. *Hydrological Sciences Journal*, 37, 217-231.

### Software Implementations

- **HBV-light**: University of Zurich (Seibert)
  <https://www.geo.uzh.ch/en/units/h2k/Services/HBV-Model.html>

- **LuMod**: UNAM (Arciniega Esparza)
  <https://gitlab.com/Zaul_AE/lumod>

- **HBV-bmi**: eWaterCycle (Daafip)
  <https://github.com/Daafip/HBV-bmi>

---

## Appendix A: Parameter Bounds Summary

### Recommended Calibration Ranges

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| TT | -2.5 | 2.5 | 0.0 | °C |
| CFMAX | 0.5 | 10.0 | 3.0 | mm/°C/d |
| CFR | 0.0 | 0.2 | 0.05 | - |
| CWH | 0.0 | 0.2 | 0.1 | - |
| SFCF | 0.4 | 1.4 | 1.0 | - |
| FC | 50 | 700 | 250 | mm |
| LP | 0.3 | 1.0 | 0.9 | - |
| BETA | 1.0 | 6.0 | 2.0 | - |
| K0 | 0.05 | 0.99 | 0.4 | 1/d |
| K1 | 0.01 | 0.5 | 0.1 | 1/d |
| K2 | 0.001 | 0.2 | 0.01 | 1/d |
| UZL | 0 | 100 | 20 | mm |
| PERC | 0 | 6 | 1.0 | mm/d |
| MAXBAS | 1 | 7 | 2.5 | d |

### Parameter Constraints

- K0 > K1 > K2 (recession rates decrease with depth)
- UZL ≥ 0 (threshold cannot be negative)
- MAXBAS ≥ 1 (at least 1 day routing)
- 0 < LP ≤ 1 (fraction of FC)
- BETA > 0 (positive exponent)

### Calibration Notes: Parameter Interactions (Equifinality)

HBV parameters exhibit significant **equifinality** - multiple parameter combinations can produce similar model performance. Key interactions to be aware of:

| Parameter Pair | Interaction |
|----------------|-------------|
| **FC and BETA** | Higher FC with lower BETA can produce similar runoff volumes to lower FC with higher BETA. Both control the soil moisture threshold behavior. |
| **K1 and K2** | These are often inversely correlated during calibration. Faster K1 with slower K2 may fit similarly to moderate values of both. |
| **PERC and K2** | Both control the lower zone dynamics. High PERC with low K2 vs. low PERC with high K2 can give similar baseflow. |
| **FC and LP** | Both affect the evapotranspiration regime. The product LP×FC determines the ET threshold. |

**Recommended Calibration Strategy:**

1. **Snow parameters first** (TT, CFMAX): Match spring flood timing
2. **Soil parameters** (FC, BETA, LP): Adjust overall water balance
3. **Response parameters** (K0, K1, K2, UZL, PERC): Fit recession curves and low flows
4. **Routing** (MAXBAS): Fine-tune peak timing

> **Warning:** Given equifinality, always report parameter uncertainty ranges rather than single "optimal" values. Consider ensemble approaches or behavioral parameter sets.

---

## Appendix B: Symbol Reference

### Primary Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| P | Precipitation | mm/d |
| T | Air temperature | °C |
| E_pot | Potential evapotranspiration | mm/d |
| E_act | Actual evapotranspiration | mm/d |
| Q | Simulated streamflow | mm/d or m³/s |

### State Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| SP | Snow pack (water equivalent) | mm |
| SM | Soil moisture storage | mm |
| SUZ | Upper zone storage | mm |
| SLZ | Lower zone storage | mm |

### Flux Variables

| Symbol | Description | Unit |
|--------|-------------|------|
| M | Snowmelt | mm/d |
| R | Refreezing | mm/d |
| I | Input to soil (rain + melt) | mm/d |
| Recharge | Flux to groundwater | mm/d |
| Q_0 | Quick/surface flow | mm/d |
| Q_1 | Interflow | mm/d |
| Q_2 | Baseflow | mm/d |
| Q_GW | Total groundwater runoff | mm/d |

### Parameters

| Symbol | Description | Unit |
|--------|-------------|------|
| TT | Temperature threshold | °C |
| CFMAX | Degree-day factor | mm/°C/d |
| CFR | Refreezing coefficient | - |
| CWH | Water holding capacity | - |
| SFCF | Snowfall correction factor | - |
| FC | Field capacity | mm |
| LP | Limit for potential ET | - |
| BETA | Shape coefficient | - |
| K0 | Quick flow recession | 1/d |
| K1 | Interflow recession | 1/d |
| K2 | Baseflow recession | 1/d |
| UZL | Upper zone threshold | mm |
| PERC | Maximum percolation | mm/d |
| MAXBAS | Routing time | d |

---

*Document compiled from analysis of Seibert & Vis (2012), hbv_hydromodel (AghaKouchak), LuMod (Arciniega Esparza), and HBV-bmi (Daafip) implementations.*
