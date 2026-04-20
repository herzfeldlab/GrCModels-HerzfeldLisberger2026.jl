module GrCModelsInterface
import ..AbstractRandomRange

export AbstractGRCBasisModel,
       init,
       model_hyperparameters,
       optimal_hyperparameters!,
       basis,
       project



# ----------------------------
# Core types
# ----------------------------

"""
Abstract supertype for all granule-cell basis models.

A concrete model type should subtype this and implement the protocol functions:
- init(model, mf_example; rng)
- basis(model, mf; warmup, warmup_mode, dt)
- hyperparams(model)
- pack(model)
- unpack!(model, θ)
"""
abstract type AbstractGRCBasisModel end


# ----------------------------
# Required protocol functions
# ----------------------------

"""
    init(model, mf, hyperparameters; rng=Random.default_rng())

Bind one-time model structure that depends on mossy-fiber dimensionality
(e.g., MF indices, mixing matrices, recurrent wiring, synapse parameters).

Concrete models MUST implement this method.
"""
function init(model::AbstractGRCBasisModel, mf, hyperparameters; rng=nothing)
    error("init(::$(typeof(model)), mf_example; rng=...) is not implemented. " *
          "Concrete models must implement init(model, mf_example, hyperparameters; rng=...).")
end


"""
    model_hyperparameters(::ModelType)

Returns a dictionary of model hyperparameters where keys are the
names of the hyperparameter and values are ranges (tuples). This function
returns the DEFAULT hyperparameters for this class of models.
"""
function model_hyperparameters(type::Type{AbstractGRCBasisModel})::Dict{Symbol, <:AbstractRandomRange}
    error("model_hyperparameters($(type)) is not implemented. " *
          "Concrete models must implement model_hyperparameters(::Type{Model}).")
end

"""
    model_hyperparameters(ModelType)

Returns a dictionary of model hyperparameters uses for the current model
implementation.
"""
function model_hyperparameters(model::AbstractGRCBasisModel)::Dict{Symbol, <:AbstractRandomRange}
    error("model_hyperparameters(::$(typeof(model))) is not implemented. " *
          "Concrete models must implement model_hyperparameters(::Type{Model}).")
end

"""
    optimal_hyperparameters(::ModelType)

Set the optimal hyperparameters for a given model. These are hard-coded based on
prior optimization. The goal is to avoid the costly task of optimizing models of
the exact same structure.
"""
function optimal_hyperparameters!(model::AbstractGRCBasisModel)
    error("optimal_hyperparameters(::$(typeof(model))) is not implemented.")
end

"""
    basis(model, mf; warmup=0, warmup_mode=:repeat_first, dt=1e-3)

Compute basis responses for MF inputs.

Input mf may be one of:
- MF x Time (AbstractMatrix)
- MF x Time x Condition (AbstractArray{<:Real,3})
- Vector{MF x Time} (AbstractVector{<:AbstractMatrix})

Return value MUST be:
- Array{Float64,3} of size (BasisUnits × Time × Condition)
  OR a Vector{Matrix} if you intentionally support ragged-time conditions.

Concrete models MUST implement this method.
"""
function basis(model::AbstractGRCBasisModel, mf; warmup::Integer=0, warmup_mode=:repeat_first, dt::Real=1e-3)
    error("basis(::$(typeof(model)), mf; warmup=..., warmup_mode=..., dt=...) is not implemented. " *
          "Concrete models must implement basis(model, mf; warmup, warmup_mode, dt).")
end


"""
    project(W, B) -> Y

Apply a linear readout to basis activity.

Inputs:
- B: basis array of size (nbasis x T x C)
- W:
  - Vector{<:Real} of length nbasis for a single output
  - Matrix{<:Real} of size (nout x nbasis) for multiple outputs

Returns:
- If W is Vector: Array{Float64,2} of size (T × C)
- If W is Matrix: Array{Float64,3} of size (nout × T × C)
"""
function project(W::AbstractVector{<:Real}, B::AbstractArray{<:Real,3})
    nb, T, C = size(B)
    length(W) == nb || error("project: length(W) must equal nbasis. Got length(W)=$(length(W)), nbasis=$nb.")
    Y = Array{Float64,2}(undef, T, C)
    for c in 1:C
        # W' * (nb×T) -> 1×T, then transpose to T×1 and drop singleton
        Y[:, c] .= (W' * B[:, :, c])'
    end
    return Y
end

function project(W::AbstractMatrix{<:Real}, B::AbstractArray{<:Real,3})
    nb, T, C = size(B)
    size(W, 2) == nb || error("project: size(W,2) must equal nbasis. Got size(W,2)=$(size(W,2)), nbasis=$nb.")
    nout = size(W, 1)
    Y = Array{Float64,3}(undef, nout, T, C)
    for c in 1:C
        Y[:, :, c] .= W * B[:, :, c]
    end
    return Y
end

end # module GrCModelsInterface