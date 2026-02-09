//! HBV-Light model flux outputs.
//!
//! `Fluxes` holds a single timestep (returned by `step()`).
//! `FluxesTimeseries` holds the full simulation (returned by `run()`).
//! `ZoneOutputs` holds per-zone details for multi-zone runs.

/// Single-timestep fluxes.
#[derive(Debug, Clone, Copy)]
pub struct Fluxes {
    pub precip: f64,
    pub temp: f64,
    pub pet: f64,
    pub precip_rain: f64,
    pub precip_snow: f64,
    pub snow_pack: f64,
    pub snow_melt: f64,
    pub liquid_water_in_snow: f64,
    pub snow_input: f64,
    pub soil_moisture: f64,
    pub recharge: f64,
    pub actual_et: f64,
    pub upper_zone: f64,
    pub lower_zone: f64,
    pub q0: f64,
    pub q1: f64,
    pub q2: f64,
    pub percolation: f64,
    pub qgw: f64,
    pub streamflow: f64,
}

/// Full timeseries of fluxes.
#[derive(Debug)]
pub struct FluxesTimeseries {
    pub precip: Vec<f64>,
    pub temp: Vec<f64>,
    pub pet: Vec<f64>,
    pub precip_rain: Vec<f64>,
    pub precip_snow: Vec<f64>,
    pub snow_pack: Vec<f64>,
    pub snow_melt: Vec<f64>,
    pub liquid_water_in_snow: Vec<f64>,
    pub snow_input: Vec<f64>,
    pub soil_moisture: Vec<f64>,
    pub recharge: Vec<f64>,
    pub actual_et: Vec<f64>,
    pub upper_zone: Vec<f64>,
    pub lower_zone: Vec<f64>,
    pub q0: Vec<f64>,
    pub q1: Vec<f64>,
    pub q2: Vec<f64>,
    pub percolation: Vec<f64>,
    pub qgw: Vec<f64>,
    pub streamflow: Vec<f64>,
}

impl FluxesTimeseries {
    /// Pre-allocate all vectors for `n` timesteps.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            precip: Vec::with_capacity(n),
            temp: Vec::with_capacity(n),
            pet: Vec::with_capacity(n),
            precip_rain: Vec::with_capacity(n),
            precip_snow: Vec::with_capacity(n),
            snow_pack: Vec::with_capacity(n),
            snow_melt: Vec::with_capacity(n),
            liquid_water_in_snow: Vec::with_capacity(n),
            snow_input: Vec::with_capacity(n),
            soil_moisture: Vec::with_capacity(n),
            recharge: Vec::with_capacity(n),
            actual_et: Vec::with_capacity(n),
            upper_zone: Vec::with_capacity(n),
            lower_zone: Vec::with_capacity(n),
            q0: Vec::with_capacity(n),
            q1: Vec::with_capacity(n),
            q2: Vec::with_capacity(n),
            percolation: Vec::with_capacity(n),
            qgw: Vec::with_capacity(n),
            streamflow: Vec::with_capacity(n),
        }
    }

    /// Push a single timestep's fluxes.
    pub fn push(&mut self, f: &Fluxes) {
        self.precip.push(f.precip);
        self.temp.push(f.temp);
        self.pet.push(f.pet);
        self.precip_rain.push(f.precip_rain);
        self.precip_snow.push(f.precip_snow);
        self.snow_pack.push(f.snow_pack);
        self.snow_melt.push(f.snow_melt);
        self.liquid_water_in_snow.push(f.liquid_water_in_snow);
        self.snow_input.push(f.snow_input);
        self.soil_moisture.push(f.soil_moisture);
        self.recharge.push(f.recharge);
        self.actual_et.push(f.actual_et);
        self.upper_zone.push(f.upper_zone);
        self.lower_zone.push(f.lower_zone);
        self.q0.push(f.q0);
        self.q1.push(f.q1);
        self.q2.push(f.q2);
        self.percolation.push(f.percolation);
        self.qgw.push(f.qgw);
        self.streamflow.push(f.streamflow);
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.streamflow.len()
    }

    /// Returns `true` if there are no timesteps.
    pub fn is_empty(&self) -> bool {
        self.streamflow.is_empty()
    }
}

/// Per-zone outputs for multi-zone runs.
///
/// 2D vectors: outer index is timestep, inner index is zone.
#[derive(Debug)]
pub struct ZoneOutputs {
    pub zone_elevations: Vec<f64>,
    pub zone_fractions: Vec<f64>,
    /// zone_temp[timestep][zone]
    pub zone_temp: Vec<Vec<f64>>,
    pub zone_precip: Vec<Vec<f64>>,
    pub snow_pack: Vec<Vec<f64>>,
    pub liquid_water_in_snow: Vec<Vec<f64>>,
    pub snow_melt: Vec<Vec<f64>>,
    pub snow_input: Vec<Vec<f64>>,
    pub soil_moisture: Vec<Vec<f64>>,
    pub recharge: Vec<Vec<f64>>,
    pub actual_et: Vec<Vec<f64>>,
}

impl ZoneOutputs {
    /// Pre-allocate for `n_timesteps` timesteps and `n_zones` zones.
    pub fn with_capacity(n_timesteps: usize, n_zones: usize) -> Self {
        let alloc_2d = || {
            let mut outer = Vec::with_capacity(n_timesteps);
            for _ in 0..n_timesteps {
                outer.push(vec![0.0; n_zones]);
            }
            outer
        };
        Self {
            zone_elevations: Vec::new(),
            zone_fractions: Vec::new(),
            zone_temp: alloc_2d(),
            zone_precip: alloc_2d(),
            snow_pack: alloc_2d(),
            liquid_water_in_snow: alloc_2d(),
            snow_melt: alloc_2d(),
            snow_input: alloc_2d(),
            soil_moisture: alloc_2d(),
            recharge: alloc_2d(),
            actual_et: alloc_2d(),
        }
    }
}
