/// GR6J — Génie Rural à 6 paramètres Journalier.
///
/// A lumped conceptual daily rainfall-runoff model with 6 parameters,
/// 3 stores, and unit hydrograph routing. Port of the airGR Fortran implementation.
pub mod constants;
pub mod fluxes;
pub mod params;
pub mod processes;
pub mod run;
pub mod state;
pub mod unit_hydrographs;
