using Random
import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange

export PassThroughModel

"""
Pass-through granule cell model.

Each granule cell averages a fixed subset of mossy fibers and applies ReLU.
This is the simplest possible basis-set model.
"""
mutable struct PassThroughModel <: AbstractGRCBasisModel
    num_granule_cells::Int # number of granule cells (basis units)
    num_mossy_fibers_per_granule_cell::Int # number of mossy fibers per granule cell
    mf_indices::Vector{Vector{Int}} # length n_grc, each length num_mossy_fibers_per_granule_cell
    mf_weights::Matrix{Float64} # Size num_granule_cells x num_mossy_fibers_per_granule_cell
    tau_m::Vector{Float64} # Membrane time constant
    bias::Vector{Float64} # Bias
    hyperparamers::Dict{Symbol, AbstractRandomRange}
end

# Default constructor
function PassThroughModel(mf, hyperparameters::Dict{Symbol, AbstractRandomRange}=model_hyperparameters(PassThroughModel);
        num_granule_cells::Int=100, 
        num_mossy_fibers_per_granule_cell::Int=4, 
        kwargs...)
    model = PassThroughModel(num_granule_cells, num_mossy_fibers_per_granule_cell, Vector{Vector{Int}}(), 
        zeros(num_granule_cells, num_mossy_fibers_per_granule_cell), zeros(num_granule_cells), zeros(num_granule_cells), hyperparameters)

    init(model, mf, hyperparameters; kwargs...)
end

function init(m::PassThroughModel, mf, hyperparameters; rng=Random.default_rng())
    MF = size(mf, 1)
    m.mf_indices = [
        rand(rng, 1:MF, m.num_mossy_fibers_per_granule_cell)
        for _ in 1:m.num_granule_cells
    ]
    m.mf_weights = rand(rng, m.num_granule_cells, m.num_mossy_fibers_per_granule_cell)
    m.tau_m = [sample(rng, Uniform(1e-3, 5e-3)) for _ in 1:m.num_granule_cells]
    m.bias = [sample(rng, hyperparameters[:bias]) for _ in 1:m.num_granule_cells]
    # Apply hyper parameters via sampling (note: no hyperparameters in this model)
    return m
end

function Base.show(io::IO, model::PassThroughModel)
    print(io, "$(typeof(model)) with $(model.num_granule_cells) granule cells.")
end

function model_hyperparameters(::Type{PassThroughModel})
    return Dict{Symbol,AbstractRandomRange}(:bias => Uniform(0.0, 150.0))
end

function model_hyperparameters(model::PassThroughModel)
    return model.hyperparamers
end

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::PassThroughModel)
    #model.hyperparamers[:bias] = Uniform(29.526626092898944, 51.50824752700273)
    model.hyperparamers[:bias] = Uniform(5.13305865035051, 5.459062196301011)
    return model.hyperparamers
end

# This is the actual driver
function basis(m::PassThroughModel, mf; warmup::Integer=0, warmup_mode=:repeat_first, dt::Real=1e-3)
    X  = normalize_mf_3d(mf)                                   # MF × T × C
    Xw = apply_warmup(X; warmup=warmup, warmup_mode=warmup_mode)

    MF, Nt, C = size(Xw)

    length(m.mf_indices) == m.num_granule_cells || error("PassThroughModel not initialized. Call init(model, mf_example).")

    B = Array{Float64,3}(undef, m.num_granule_cells, Nt, C)

    for i in 1:m.num_granule_cells
        I = m.mf_indices[i]
        w = m.mf_weights[i, :]
        for c in 1:C
            drive = dropdims(sum(Xw[I, :, c] .* w, dims=1), dims=1) .- m.bias[i]
            rate = drive[1]
            v = drive[1]
            for t = 1:Nt
                v += dt * ((-v + drive[t]) / m.tau_m[i])
                B[i, t, c] = v
            end
            #B[i, :, c] .= dropdims(sum(Xw[I, :, c] .* w, dims=1), dims=1)
        end
    end

    @. B = max(B, 0.0)

    return warmup > 0 ? B[:, warmup+1:end, :] : B
end
