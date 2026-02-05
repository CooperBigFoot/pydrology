# ForcingData Validation Contract

This document specifies the validation rules and data contract for the `ForcingData` class.

## Overview

`ForcingData` is a Pydantic-validated container for hydrological forcing data. It enforces
strict validation rules to ensure data quality and prevent runtime errors.

## Fields

| Field | Type | Required | Units | Description |
|-------|------|----------|-------|-------------|
| `time` | `np.ndarray[datetime64]` | Yes | - | Timestamp for each observation |
| `precip` | `np.ndarray[float64]` | Yes | mm/day | Daily precipitation |
| `pet` | `np.ndarray[float64]` | Yes | mm/day | Daily potential evapotranspiration |
| `temp` | `np.ndarray[float64]` | No* | deg C | Daily mean air temperature |
| `resolution` | `Resolution` | No | - | Temporal resolution (default: `Resolution.daily`) |

*Required when using the GR6J-CemaNeige coupled model (`pydrology.models.gr6j_cemaneige`)

### Resolution Enum

The `Resolution` enum specifies the temporal resolution of the forcing data:

| Value | Description | Typical Time Spacing |
|-------|-------------|---------------------|
| `Resolution.hourly` | Hourly data | ~1 hour |
| `Resolution.daily` | Daily data (default) | 22-26 hours |
| `Resolution.monthly` | Monthly data | 27-32 days |
| `Resolution.annual` | Annual data | 360-370 days |

```python
from pydrology import Resolution, ForcingData

# Daily forcing (default)
forcing = ForcingData(
    time=time_array,
    precip=precip_array,
    pet=pet_array,
    resolution=Resolution.daily,  # Optional, this is the default
)

# Monthly forcing
monthly_forcing = ForcingData(
    time=monthly_time_array,
    precip=monthly_precip_array,
    pet=monthly_pet_array,
    resolution=Resolution.monthly,
)
```

## Validation Rules

### 1. Array Dimensionality

All arrays must be 1-dimensional.

**Valid:**

```python
precip = np.array([1.0, 2.0, 3.0])  # Shape: (3,)
```

**Invalid:**

```python
precip = np.array([[1.0], [2.0]])   # Shape: (2, 1) - 2D array
```

**Error:** `ValidationError: precip array must be 1D, got 2D`

### 2. Length Consistency

All provided arrays must have the same length as the `time` array.

**Valid:**

```python
time = np.array(['2020-01-01', '2020-01-02'], dtype='datetime64')
precip = np.array([1.0, 2.0])  # Same length as time
```

**Invalid:**

```python
time = np.array(['2020-01-01', '2020-01-02'], dtype='datetime64')
precip = np.array([1.0, 2.0, 3.0])  # Different length
```

**Error:** `ValidationError: precip length 3 does not match time length 2`

### 3. NaN Rejection

NaN values are not allowed in any numeric array (precip, pet, temp).

**Invalid:**

```python
precip = np.array([1.0, np.nan, 3.0])
```

**Error:** `ValidationError: precip array contains NaN values`

**Why:** NaN values in forcing data typically indicate data quality issues. The model
requires explicit handling of missing data before simulation.

### 4. Resolution Validation

When creating ForcingData, the time spacing is validated against the specified resolution:

**Valid:**

```python
# Daily data with ~24 hour spacing
time = np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[D]')
forcing = ForcingData(time=time, precip=precip, pet=pet, resolution=Resolution.daily)
```

**Invalid:**

```python
# Monthly data but resolution=daily
monthly_time = np.array(['2020-01-01', '2020-02-01', '2020-03-01'], dtype='datetime64[D]')
forcing = ForcingData(time=monthly_time, precip=precip, pet=pet, resolution=Resolution.daily)
```

**Error:** `ValidationError: Time spacing (median 744.0 hours) does not match resolution 'daily' (expected 22.0-26.0 hours)`

### 5. Type Coercion

**Numeric arrays:** Automatically coerced to `float64`

```python
# Input as integers
precip = np.array([1, 2, 3])  # int64
# Stored as float64
assert forcing.precip.dtype == np.float64
```

**Time array:** Automatically coerced to `datetime64[ns]`

```python
# Input as strings or datetime objects
time = np.array(['2020-01-01', '2020-01-02'])
# Stored as datetime64[ns]
assert forcing.time.dtype == np.dtype('datetime64[ns]')
```

## Immutability

ForcingData is frozen (immutable) after creation:

```python
forcing = ForcingData(...)
forcing.precip = new_array  # Raises ValidationError
```

## Creating ForcingData

### From NumPy Arrays

```python
import numpy as np
from pydrology import ForcingData

forcing = ForcingData(
    time=np.arange(365, dtype='datetime64[D]') + np.datetime64('2020-01-01'),
    precip=np.random.uniform(0, 20, 365),
    pet=np.random.uniform(1, 5, 365),
)
```

### From Pandas DataFrame

```python
import pandas as pd
from pydrology import ForcingData

df = pd.read_csv('forcing.csv', parse_dates=['date'])

forcing = ForcingData(
    time=df['date'].values,
    precip=df['precip'].values,
    pet=df['pet'].values,
    temp=df['temp'].values if 'temp' in df else None,
)
```

### From CSV with pandas

```python
import pandas as pd
from pydrology import ForcingData

df = pd.read_csv('forcing.csv')

forcing = ForcingData(
    time=pd.to_datetime(df['date']).values,
    precip=df['precip'].to_numpy(),
    pet=df['pet'].to_numpy(),
)
```

### From NetCDF

```python
import xarray as xr
from pydrology import ForcingData

ds = xr.open_dataset('forcing.nc')

forcing = ForcingData(
    time=ds['time'].values,
    precip=ds['precipitation'].values,
    pet=ds['pet'].values,
)
```

## Error Handling

### ValidationError Structure

Pydantic validation errors contain detailed information:

```python
try:
    forcing = ForcingData(
        time=np.arange(3, dtype='datetime64[D]'),
        precip=np.array([1.0, np.nan, 3.0]),
        pet=np.array([1.0, 2.0, 3.0]),
    )
except ValidationError as e:
    print(e.errors())
    # [{'loc': ('precip',), 'msg': 'precip array contains NaN values', ...}]
```

### Handling Missing Data

Before creating ForcingData, handle missing values:

```python
import numpy as np
import pandas as pd

# Option 1: Forward fill
df['precip'] = df['precip'].ffill()

# Option 2: Interpolate
df['pet'] = df['pet'].interpolate()

# Option 3: Remove rows with missing values
df = df.dropna(subset=['precip', 'pet'])

# Now create ForcingData
forcing = ForcingData(
    time=df['date'].values,
    precip=df['precip'].values,
    pet=df['pet'].values,
)
```

## Temporal Aggregation

ForcingData provides an `aggregate()` method for coarsening data to a lower temporal resolution (e.g., daily to monthly). This is useful when you have high-resolution data but want to run models at coarser timesteps.

### aggregate() Method

```python
def aggregate(
    self,
    target: Resolution,
    methods: dict[str, str] | None = None,
) -> ForcingData:
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `target` | `Resolution` | required | Target resolution (must be coarser than current) |
| `methods` | `dict[str, str]` | `None` | Aggregation method per field |

**Default Aggregation Methods:**

| Field | Default Method | Rationale |
|-------|---------------|-----------|
| `precip` | `sum` | Total precipitation over period |
| `pet` | `sum` | Total potential ET over period |
| `temp` | `mean` | Average temperature over period |

**Returns:** New `ForcingData` instance at the target resolution.

**Raises:**
- `ValueError`: If target is not coarser than current resolution
- `ImportError`: If polars is not installed

### Usage Examples

**Daily to Monthly:**

```python
from pydrology import Resolution, ForcingData

# Daily forcing data
daily_forcing = ForcingData(
    time=daily_time,
    precip=daily_precip,
    pet=daily_pet,
    temp=daily_temp,
    resolution=Resolution.daily,
)

# Aggregate to monthly (requires polars: uv add polars)
monthly_forcing = daily_forcing.aggregate(Resolution.monthly)
print(len(monthly_forcing))  # ~12 timesteps per year
```

**Custom Aggregation Methods:**

```python
# Use mean instead of sum for precipitation (e.g., for intensity analysis)
monthly_forcing = daily_forcing.aggregate(
    Resolution.monthly,
    methods={"precip": "mean", "pet": "sum", "temp": "mean"},
)
```

**Hourly to Daily:**

```python
hourly_forcing = ForcingData(
    time=hourly_time,
    precip=hourly_precip,
    pet=hourly_pet,
    resolution=Resolution.hourly,
)
daily_forcing = hourly_forcing.aggregate(Resolution.daily)
```

### Restrictions

- **Coarsening only:** You can only aggregate to a coarser resolution. Attempting to aggregate from monthly to daily will raise a `ValueError`.
- **Supported targets:** Aggregation supports `Resolution.daily`, `Resolution.monthly`, and `Resolution.annual` as targets.
- **Requires polars:** The aggregation uses polars for efficient time-based grouping. Install with `uv add polars`.

```python
# This will raise ValueError
monthly_forcing.aggregate(Resolution.daily)  # Cannot refine resolution
```

## Best Practices

1. **Validate early:** Create ForcingData as early as possible in your pipeline to catch
   data issues immediately.

2. **Check lengths:** If loading from multiple sources, verify array lengths match before
   creating ForcingData.

3. **Handle NaN explicitly:** Decide on a strategy for missing data (interpolation,
   fill, or exclusion) before creating ForcingData.

4. **Use appropriate dtypes:** While coercion is automatic, providing data in the correct
   dtype (float64 for numerics, datetime64 for time) is more efficient.

5. **Match resolution to model:** Ensure your forcing data resolution is supported by your
   model. Check `model.SUPPORTED_RESOLUTIONS` to see which resolutions a model accepts.
