using Random
import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange,
    init,
    basis,
    model_hyperparameters

import ..Uniform, ..sample

export NonLinearPassThroughModel

mutable struct SigmoidActivation
    gain::Float64   # slope
    bias::Float64   # horizontal shift
end
SigmoidActivation(; gain=1.0, bias=0.0) = SigmoidActivation(gain, bias)
apply_activation(a::SigmoidActivation, x) = @. 1.0 / (1.0 + exp(-(a.gain * x - a.bias)))

"""
A nonlinear pass-through granule cell model (via non-linear activation function)

Each granule cell averages a fixed subset of mossy fibers and applies a sigmoid
activation funciton (with two free parameters: the bias and the gain).
"""
mutable struct NonLinearPassThroughModel <: AbstractGRCBasisModel
    num_granule_cells::Int # number of granule cells (basis units)
    num_mossy_fibers_per_granule_cell::Int # number of mossy fibers per granule cell
    mf_indices::Vector{Vector{Int}} # length n_grc, each length num_mossy_fibers_per_granule_cell
    mf_weights::Matrix{Float64} # Size num_granule_cells x num_mossy_fibers_per_granule_cell
    activation_functions::Vector{SigmoidActivation}
    tau_m::Vector{Float64} # Membrane time constants
    hyperparameters::Dict{Symbol, AbstractRandomRange}
end

# Default constructor
function NonLinearPassThroughModel(mf, hyperparameters::Dict{Symbol, AbstractRandomRange}=model_hyperparameters(NonLinearPassThroughModel);
        num_granule_cells::Int=100, 
        num_mossy_fibers_per_granule_cell::Int=4, 
        kwargs...)
    model = NonLinearPassThroughModel(num_granule_cells, num_mossy_fibers_per_granule_cell, Vector{Vector{Int}}(), 
        zeros(num_granule_cells, num_mossy_fibers_per_granule_cell), 
        [SigmoidActivation() for _ in 1:num_granule_cells],
        zeros(num_granule_cells),
        hyperparameters)
    init(model, mf, hyperparameters; kwargs...)
end

function init(m::NonLinearPassThroughModel, mf, hyperparameters::Dict{Symbol, AbstractRandomRange}; rng=Random.default_rng())
    MF = size(mf, 1)
    m.mf_indices = [
        rand(rng, 1:MF, m.num_mossy_fibers_per_granule_cell)
        for _ in 1:m.num_granule_cells
    ]
    m.hyperparameters = hyperparameters
    m.mf_weights = rand(rng, m.num_granule_cells, m.num_mossy_fibers_per_granule_cell)
    for i = 1:m.num_granule_cells
        gain = sample(rng, m.hyperparameters[:gain])
        bias = sample(rng, m.hyperparameters[:bias])
        m.activation_functions[i] = SigmoidActivation(gain, bias)
        m.tau_m[i] = sample(rng, Uniform(1e-3, 5e-3))
    end

    # Apply hyper parameters via sampling (note: no hyperparameters in this model)
    return m
end

function Base.show(io::IO, model::NonLinearPassThroughModel)
    print(io, "$(typeof(model)) with $(model.num_granule_cells) granule cells.")
end

function model_hyperparameters(::Type{NonLinearPassThroughModel})
    # No hyperparameters for this model
    return Dict{Symbol,AbstractRandomRange}(
        :gain => Uniform(0, 1), 
        :bias => Uniform(0, 150)
    )
end

function model_hyperparameters(model::NonLinearPassThroughModel)
    # No hyperparameters for this model
    return model.hyperparameters
end

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::NonLinearPassThroughModel)
    model.hyperparameters[:bias] = Uniform(5.442145079042633, 20.178139289408875)
    model.hyperparameters[:gain] = GrCModels.Uniform(0.7706314156625467, 0.9982626628217095)
    #hyp[:bias] = GrCModels.Uniform(5.442145079042633, 20.178139289408875)
    #hyp[:gain] = GrCModels.Uniform(0.7706314156625467, 0.9982626628217095)

    return model.hyperparameters
end

# This is the actual driver
function basis(m::NonLinearPassThroughModel, mf; warmup::Integer=0, warmup_mode=:repeat_first, dt::Real=1e-3)
    X  = normalize_mf_3d(mf)                                   # MF × T × C
    Xw = apply_warmup(X; warmup=warmup, warmup_mode=warmup_mode)

    MF, Nt, C = size(Xw)

    length(m.mf_indices) == m.num_granule_cells || error("PassThroughModel not initialized. Call init(model, mf_example).")

    B = Array{Float64,3}(undef, m.num_granule_cells, Nt, C)
    for i in 1:m.num_granule_cells
        I = m.mf_indices[i]
        w = m.mf_weights[i, :]
        for c in 1:C
            drive = dropdims(sum(Xw[I, :, c] .* w, dims=1), dims=1)
            v = drive[1]
            for t = 1:Nt
                v += dt * ((-v + drive[t]) / m.tau_m[i])
                B[i, t, c] = apply_activation(m.activation_functions[i], v)
            end
        end
    end

    return warmup > 0 ? B[:, warmup+1:end, :] : B
end
