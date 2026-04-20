# GrCModels.jl
A small Julia package that standardizes **granule-cell (GrC) basis-set models** behind a single interface 
so you can run **fitting** and **hyperparameter searches**,
 with shared code across heterogeneous model implementations.

The core idea: every model consumes mossy-fiber (MF) inputs and 
returns a **basis matrix over time** with a consistent shape, 
plus a standardized way to expose tunable hyperparameters and 
perform linear projections onto one or more outputs (e.g., Purkinje cell fits).

## Interface overview

All models implement the `AbstractGRCBasisModel` protocol.

### 1) Initialization / binding random structure

Many models require one-time book-keeping, like keeping track of the number of mossy fiber inputs
and granule cells int he overall model. This is performed by `init`

```julia
model = init(model, mf_example; rng=Random.default_rng())
```

In addition, the key function for initialization is `resample!`. This function serves to randomly
create a model by drawing hyperparameters from the range of their distributions. 

### 2) Basis computation
Compute basis responses for MF inputs with optional warmup. Inputs may be:

- MF × Time matrix  
- MF × Time × Condition array  

The canonical output is always:

- B::Array{Float64,3} with shape (BasisUnits × Time × Condition)

```julia
B = basis(model, mf; warmup=0, warmup_mode=:repeat_first, dt=1e-3)
```

Warmup adds pre-roll samples to stabilize stateful dynamics and is removed from the returned basis by default.

### 3) Hyperparameter exposure (PSO-friendly)
Models advertise tunable hyperparameters via bounded specifications and a consistent pack/unpack protocol.

```julia
specs = hyperparameters(model)
```

This lets you plug any model into the same optimizer.

## Quick start

```julia
using Random
using GrCModels

mf = randn(300, 2000)
model = GilmerModel(mf)

# Then fit weights to Purkinje cell responses
pc = randn(300, 2000)
w, y_hat = fit_purkinje_cell_responses(model, mf, pc)
```
---

## Included models
GrCModels.jl provides adapters that conform to the interface for the following model families:
1. `PassThroughModel`  
   Baseline: GrC inputs are simple MF mixing with rectification.
2. `NonLinearPassThroughModel`
   Weighted MF activity is passed through a sigmoid activation function.
2. `GilmerModel`  
   Randomly selects N mossy fibers per GrC, averages them, thresholds by mean + z·std, then applies ReLU.
3. `RandomProjectionModel`
   Random MF mixing followed by recurrent inhibition with an exponential kernel (τ) and recurrent weight scaling (κ).
4. `DifferentiatorUnitModel` 
   Herzfeld & Lisberger–style (2025, BioRxiv) differentiator units using a mixing matrix and temporal filtering dynamics.
6. STPModel  
   Short-term plasticity synapse model driving GrC dynamics.

## Hyperparameter optimization
Typical workflow:
1. Create model and init it once.
2. Call `optimize_hyperparameters(model, mf, pc)`

## Data Availability
Data from Herzfeld & Lisberger (2026) is available on OSF (Open Science Framework). These data are referenced
in the attached notebooks. In particular, trial averaged data from mossy fibers, unipolar brush cells, Purkinje
cells, and molecular layer inteneurons during smooth pursuit are useful additions to the code in this repository.