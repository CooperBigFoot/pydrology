use pydrology_macros::Fluxes;

#[derive(Debug, Clone, Fluxes)]
pub struct BadFluxes {
    pub pet: f64,
    pub count: u32,
}

fn main() {}
