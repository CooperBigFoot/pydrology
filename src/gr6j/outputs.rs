//! GR6J model flux outputs.
//!
//! Two levels: `Fluxes` holds a single timestep, `FluxesTimeseries` holds
//! the full simulation (Vec of each field). Mirrors the airGR MISC array (1-20).

/// Single-timestep fluxes — returned by `step()`.
#[derive(Debug, Clone, Copy)]
pub struct Fluxes {
    pub pet: f64,
    pub precip: f64,
    pub production_store: f64,
    pub net_rainfall: f64,
    pub storage_infiltration: f64,
    pub actual_et: f64,
    pub percolation: f64,
    pub effective_rainfall: f64,
    pub q9: f64,
    pub q1: f64,
    pub routing_store: f64,
    pub exchange: f64,
    pub actual_exchange_routing: f64,
    pub actual_exchange_direct: f64,
    pub actual_exchange_total: f64,
    pub qr: f64,
    pub qrexp: f64,
    pub exponential_store: f64,
    pub qd: f64,
    pub streamflow: f64,
}

/// Full timeseries of fluxes — returned by `run()`.
#[derive(Debug)]
pub struct FluxesTimeseries {
    pub pet: Vec<f64>,
    pub precip: Vec<f64>,
    pub production_store: Vec<f64>,
    pub net_rainfall: Vec<f64>,
    pub storage_infiltration: Vec<f64>,
    pub actual_et: Vec<f64>,
    pub percolation: Vec<f64>,
    pub effective_rainfall: Vec<f64>,
    pub q9: Vec<f64>,
    pub q1: Vec<f64>,
    pub routing_store: Vec<f64>,
    pub exchange: Vec<f64>,
    pub actual_exchange_routing: Vec<f64>,
    pub actual_exchange_direct: Vec<f64>,
    pub actual_exchange_total: Vec<f64>,
    pub qr: Vec<f64>,
    pub qrexp: Vec<f64>,
    pub exponential_store: Vec<f64>,
    pub qd: Vec<f64>,
    pub streamflow: Vec<f64>,
}

impl FluxesTimeseries {
    /// Pre-allocate all vectors for `n` timesteps.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            pet: Vec::with_capacity(n),
            precip: Vec::with_capacity(n),
            production_store: Vec::with_capacity(n),
            net_rainfall: Vec::with_capacity(n),
            storage_infiltration: Vec::with_capacity(n),
            actual_et: Vec::with_capacity(n),
            percolation: Vec::with_capacity(n),
            effective_rainfall: Vec::with_capacity(n),
            q9: Vec::with_capacity(n),
            q1: Vec::with_capacity(n),
            routing_store: Vec::with_capacity(n),
            exchange: Vec::with_capacity(n),
            actual_exchange_routing: Vec::with_capacity(n),
            actual_exchange_direct: Vec::with_capacity(n),
            actual_exchange_total: Vec::with_capacity(n),
            qr: Vec::with_capacity(n),
            qrexp: Vec::with_capacity(n),
            exponential_store: Vec::with_capacity(n),
            qd: Vec::with_capacity(n),
            streamflow: Vec::with_capacity(n),
        }
    }

    /// Push a single timestep's fluxes into the timeseries.
    pub fn push(&mut self, f: &Fluxes) {
        self.pet.push(f.pet);
        self.precip.push(f.precip);
        self.production_store.push(f.production_store);
        self.net_rainfall.push(f.net_rainfall);
        self.storage_infiltration.push(f.storage_infiltration);
        self.actual_et.push(f.actual_et);
        self.percolation.push(f.percolation);
        self.effective_rainfall.push(f.effective_rainfall);
        self.q9.push(f.q9);
        self.q1.push(f.q1);
        self.routing_store.push(f.routing_store);
        self.exchange.push(f.exchange);
        self.actual_exchange_routing.push(f.actual_exchange_routing);
        self.actual_exchange_direct.push(f.actual_exchange_direct);
        self.actual_exchange_total.push(f.actual_exchange_total);
        self.qr.push(f.qr);
        self.qrexp.push(f.qrexp);
        self.exponential_store.push(f.exponential_store);
        self.qd.push(f.qd);
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
