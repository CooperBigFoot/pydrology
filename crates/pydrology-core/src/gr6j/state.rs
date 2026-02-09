/// GR6J model state variables.
///
/// Mutable state that evolves during simulation. Contains three stores
/// and unit hydrograph convolution states:
/// - `production_store`: S — soil moisture store level [mm]
/// - `routing_store`: R — groundwater/routing store level [mm]
/// - `exponential_store`: Exp — slow drainage store, can be negative [mm]
/// - `uh1_states`: Convolution states for UH1 (20 elements)
/// - `uh2_states`: Convolution states for UH2 (40 elements)
use super::constants::{NH, STATE_SIZE};
use super::params::Parameters;
use crate::traits::ModelState;

#[derive(Debug, Clone)]
pub struct State {
    pub production_store: f64,
    pub routing_store: f64,
    pub exponential_store: f64,
    pub uh1_states: [f64; NH],
    pub uh2_states: [f64; 2 * NH],
}

impl State {
    /// Create initial state from parameters.
    ///
    /// Uses standard initialization fractions:
    /// - Production store at 30% of X1 capacity
    /// - Routing store at 50% of X3 capacity
    /// - Exponential store at zero
    /// - Unit hydrograph states all zero
    pub fn initialize(params: &Parameters) -> Self {
        Self {
            production_store: 0.3 * params.x1,
            routing_store: 0.5 * params.x3,
            exponential_store: 0.0,
            uh1_states: [0.0; NH],
            uh2_states: [0.0; 2 * NH],
        }
    }

    /// Convert state to a flat array of 63 elements.
    ///
    /// Layout: [production_store, routing_store, exponential_store, uh1_states[0:20], uh2_states[0:40]]
    pub fn to_array(&self) -> [f64; 63] {
        let mut arr = [0.0; 63];
        arr[0] = self.production_store;
        arr[1] = self.routing_store;
        arr[2] = self.exponential_store;
        arr[3..23].copy_from_slice(&self.uh1_states);
        arr[23..63].copy_from_slice(&self.uh2_states);
        arr
    }

    /// Reconstruct State from a flat array of 63 elements.
    pub fn from_array(arr: &[f64; 63]) -> Self {
        let mut uh1_states = [0.0; NH];
        let mut uh2_states = [0.0; 2 * NH];
        uh1_states.copy_from_slice(&arr[3..23]);
        uh2_states.copy_from_slice(&arr[23..63]);
        Self {
            production_store: arr[0],
            routing_store: arr[1],
            exponential_store: arr[2],
            uh1_states,
            uh2_states,
        }
    }
}

impl ModelState for State {
    fn to_vec(&self) -> Vec<f64> {
        self.to_array().to_vec()
    }

    fn from_slice(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != STATE_SIZE {
            return Err(format!(
                "expected {} state elements, got {}",
                STATE_SIZE,
                arr.len()
            ));
        }
        let mut fixed = [0.0f64; 63];
        fixed.copy_from_slice(arr);
        Ok(Self::from_array(&fixed))
    }

    fn array_len(&self) -> usize {
        STATE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> Parameters {
        Parameters::new(350.0, 0.0, 90.0, 1.7, 0.0, 5.0)
    }

    #[test]
    fn initialize_production_store() {
        let p = test_params();
        let s = State::initialize(&p);
        assert_eq!(s.production_store, 0.3 * 350.0);
    }

    #[test]
    fn initialize_routing_store() {
        let p = test_params();
        let s = State::initialize(&p);
        assert_eq!(s.routing_store, 0.5 * 90.0);
    }

    #[test]
    fn initialize_exponential_store() {
        let p = test_params();
        let s = State::initialize(&p);
        assert_eq!(s.exponential_store, 0.0);
    }

    #[test]
    fn initialize_uh_states_zero() {
        let p = test_params();
        let s = State::initialize(&p);
        assert!(s.uh1_states.iter().all(|&v| v == 0.0));
        assert!(s.uh2_states.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn roundtrip_array_conversion() {
        let p = test_params();
        let s = State::initialize(&p);
        let arr = s.to_array();
        let s2 = State::from_array(&arr);
        assert_eq!(s.production_store, s2.production_store);
        assert_eq!(s.routing_store, s2.routing_store);
        assert_eq!(s.exponential_store, s2.exponential_store);
        assert_eq!(s.uh1_states, s2.uh1_states);
        assert_eq!(s.uh2_states, s2.uh2_states);
    }

    #[test]
    fn model_state_roundtrip() {
        use crate::traits::ModelState;

        let p = test_params();
        let s = State::initialize(&p);
        let v = s.to_vec();
        let s2 = State::from_slice(&v).unwrap();
        assert_eq!(s.production_store, s2.production_store);
        assert_eq!(s.routing_store, s2.routing_store);
        assert_eq!(s.exponential_store, s2.exponential_store);
        assert_eq!(s.uh1_states, s2.uh1_states);
        assert_eq!(s.uh2_states, s2.uh2_states);
    }

    #[test]
    fn model_state_wrong_length() {
        use crate::traits::ModelState;

        assert!(State::from_slice(&[1.0, 2.0]).is_err());
    }
}
