/// Core trait for lumped hydrological models.
///
/// Defines the interface all single-zone models implement: prepare context,
/// initialize state, step, and run over a timeseries.
pub trait HydrologicalModel {
    type Params;
    type State: Clone;
    type Forcing: Copy;
    type Fluxes;
    type FluxesTimeseries: FluxesTimeseriesOps<Self::Fluxes>;
    /// Precomputed context derived from params, constant for a given run.
    type Context;

    /// Precompute any run-constant data from parameters (e.g., UH ordinates).
    fn prepare(params: &Self::Params) -> Self::Context;

    /// Create a default initial state from parameters.
    fn initialize_state(params: &Self::Params) -> Self::State;

    /// Execute one timestep: given state, params, forcing, and context,
    /// return the new state and fluxes.
    fn step(
        state: &Self::State,
        params: &Self::Params,
        forcing: &Self::Forcing,
        context: &Self::Context,
    ) -> (Self::State, Self::Fluxes);

    /// Run the model over a forcing timeseries.
    ///
    /// Default implementation: prepare context, initialize/use provided state,
    /// loop over forcing calling step.
    fn run(
        params: &Self::Params,
        forcing: &[Self::Forcing],
        initial_state: Option<&Self::State>,
    ) -> Self::FluxesTimeseries {
        let context = Self::prepare(params);
        let mut state = match initial_state {
            Some(s) => s.clone(),
            None => Self::initialize_state(params),
        };

        let n = forcing.len();
        let mut outputs = Self::FluxesTimeseries::with_capacity(n);

        for f in forcing {
            let (new_state, fluxes) = Self::step(&state, params, f, &context);
            outputs.push(&fluxes);
            state = new_state;
        }

        outputs
    }
}

/// Operations required on the timeseries collection type.
pub trait FluxesTimeseriesOps<F> {
    fn with_capacity(n: usize) -> Self;
    fn push(&mut self, f: &F);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
