/// GR2M model state variables.
///
/// Mutable state that evolves during simulation. Contains the two stores:
/// - `production_store`: S — soil moisture store level [mm]
/// - `routing_store`: R — groundwater/routing store level [mm]
use super::params::Parameters;

#[derive(Debug, Clone, Copy)]
pub struct State {
    pub production_store: f64,
    pub routing_store: f64,
}

impl State {
    /// Create initial state from parameters.
    ///
    /// Uses standard initialization fractions:
    /// - Production store at 30% of X1 capacity
    /// - Routing store at 30% of X1
    pub fn initialize(params: &Parameters) -> Self {
        Self {
            production_store: 0.3 * params.x1,
            routing_store: 0.3 * params.x1,
        }
    }
}
