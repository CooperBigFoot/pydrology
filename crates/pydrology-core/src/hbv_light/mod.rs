/// HBV-Light -- a lumped conceptual rainfall-runoff model.
///
/// A semi-distributed model with 14 parameters, supporting single-zone
/// and multi-zone (elevation band) operation. Includes snow routine,
/// soil moisture routine, response routine, and triangular UH routing.
pub mod constants;
pub mod fluxes;
pub mod params;
pub mod processes;
pub mod routing;
pub mod run;
pub mod state;
