/// GR2M model state variables.
///
/// Mutable state that evolves during simulation. Contains the two stores:
/// - `production_store`: S — soil moisture store level [mm]
/// - `routing_store`: R — groundwater/routing store level [mm]
use super::constants::STATE_SIZE;
use super::params::Parameters;
use crate::traits::ModelState;

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

impl ModelState for State {
    fn to_vec(&self) -> Vec<f64> {
        vec![self.production_store, self.routing_store]
    }

    fn from_slice(arr: &[f64]) -> Result<Self, String> {
        if arr.len() != STATE_SIZE {
            return Err(format!(
                "expected {} state elements, got {}",
                STATE_SIZE,
                arr.len()
            ));
        }
        Ok(Self {
            production_store: arr[0],
            routing_store: arr[1],
        })
    }

    fn array_len(&self) -> usize {
        STATE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ModelState;

    fn test_params() -> Parameters {
        Parameters::new(500.0, 1.0).unwrap()
    }

    #[test]
    fn to_vec_from_slice_roundtrip() {
        let p = test_params();
        let s = State::initialize(&p);
        let v = s.to_vec();
        let s2 = State::from_slice(&v).unwrap();
        assert_eq!(s.production_store, s2.production_store);
        assert_eq!(s.routing_store, s2.routing_store);
    }

    #[test]
    fn from_slice_wrong_length() {
        assert!(State::from_slice(&[1.0]).is_err());
        assert!(State::from_slice(&[1.0, 2.0, 3.0]).is_err());
    }
}
