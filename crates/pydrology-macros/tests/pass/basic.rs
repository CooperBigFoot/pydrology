use pydrology_macros::Fluxes;

#[derive(Debug, Clone, Copy, Fluxes)]
pub struct TestFluxes {
    pub pet: f64,
    pub precip: f64,
    pub streamflow: f64,
}

fn main() {
    let f = TestFluxes { pet: 1.0, precip: 2.0, streamflow: 3.0 };
    let mut ts = TestFluxesTimeseries::with_capacity(10);
    ts.push(&f);
    assert_eq!(ts.len(), 1);
    assert!(!ts.is_empty());
    assert_eq!(TestFluxes::field_names(), &["pet", "precip", "streamflow"]);
}
