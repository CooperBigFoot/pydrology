//! CemaNeige snow accumulation and melt model.
//!
//! A degree-day snow model with thermal state tracking, used standalone
//! or coupled with GR6J for cold-climate catchment simulation.

pub mod constants;
pub mod coupled;
pub mod params;
pub mod processes;
pub mod run;
pub mod state;
