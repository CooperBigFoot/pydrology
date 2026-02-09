/// GR2M model flux outputs.
///
/// Two levels: `Fluxes` holds a single timestep, `FluxesTimeseries` holds
/// the full simulation (Vec of each field). Mirrors the airGR MISC array (1-11).
///
/// Single-timestep fluxes — returned by `step()`.
#[derive(Debug, Clone, Copy)]
pub struct Fluxes {
    pub pet: f64,              // MISC(1)  — potential ET [mm/month]
    pub precip: f64,           // MISC(2)  — precipitation [mm/month]
    pub production_store: f64, // MISC(3)  — store level after timestep [mm]
    pub rainfall_excess: f64,  // MISC(4)  — P1 [mm/month]
    pub storage_fill: f64,     // MISC(5)  — PS [mm/month]
    pub actual_et: f64,        // MISC(6)  — AE [mm/month]
    pub percolation: f64,      // MISC(7)  — P2 [mm/month]
    pub routing_input: f64,    // MISC(8)  — P3 = P1 + P2 [mm/month]
    pub routing_store: f64,    // MISC(9)  — routing store after timestep [mm]
    pub exchange: f64,         // MISC(10) — AEXCH [mm/month]
    pub streamflow: f64,       // MISC(11) — Q [mm/month]
}

/// Full timeseries of fluxes — returned by `run()`.
#[derive(Debug)]
pub struct FluxesTimeseries {
    pub pet: Vec<f64>,
    pub precip: Vec<f64>,
    pub production_store: Vec<f64>,
    pub rainfall_excess: Vec<f64>,
    pub storage_fill: Vec<f64>,
    pub actual_et: Vec<f64>,
    pub percolation: Vec<f64>,
    pub routing_input: Vec<f64>,
    pub routing_store: Vec<f64>,
    pub exchange: Vec<f64>,
    pub streamflow: Vec<f64>,
}

impl FluxesTimeseries {
    /// Pre-allocate all vectors for `n` timesteps.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            pet: Vec::with_capacity(n),
            precip: Vec::with_capacity(n),
            production_store: Vec::with_capacity(n),
            rainfall_excess: Vec::with_capacity(n),
            storage_fill: Vec::with_capacity(n),
            actual_et: Vec::with_capacity(n),
            percolation: Vec::with_capacity(n),
            routing_input: Vec::with_capacity(n),
            routing_store: Vec::with_capacity(n),
            exchange: Vec::with_capacity(n),
            streamflow: Vec::with_capacity(n),
        }
    }

    /// Push a single timestep's fluxes into the timeseries.
    pub fn push(&mut self, f: &Fluxes) {
        self.pet.push(f.pet);
        self.precip.push(f.precip);
        self.production_store.push(f.production_store);
        self.rainfall_excess.push(f.rainfall_excess);
        self.storage_fill.push(f.storage_fill);
        self.actual_et.push(f.actual_et);
        self.percolation.push(f.percolation);
        self.routing_input.push(f.routing_input);
        self.routing_store.push(f.routing_store);
        self.exchange.push(f.exchange);
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
