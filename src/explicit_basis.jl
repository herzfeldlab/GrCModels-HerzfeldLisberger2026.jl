using Random
import LinearAlgebra: I

import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange

import ..Uniform, ..sample

export DifferentiatorUnitModel

"""
    DifferentiatorUnitModel

A dynamical system that takes a single input and performs differentiation
per Herzfeld & Lisberger (2025, BioRxiv). The differentiator is modeled as a two-state
system of the following form 
 dx = A x + b (u - bias)
 y = σ(c x)
The system is defined by is given by
A = [alpha 1 ; -k^2 -2k]
and 
b = [0 k^2].

In Herzfeld & Lisberger, we assumed that σ was an ReLu activator function,
ensuring the output of the system was always positive and c = [0 slope], where
the output passed through the ReLu corresponded to the second state.

In this dynamical system, there are three free parameters. The parameter alpha controls
the magnitude of low-pass information that is passed without differentiator to the output.
The parameter k control the speed of differentiation. Finally, the bias represents
net inhibition that reduces the number of active differentiator units in the system
for a given input.
"""
mutable struct DifferentiatorUnitModel <: AbstractGRCBasisModel
    num_granule_cells::Int
    num_mossy_fibers_per_granule_cell::Int
    mf_indices::Vector{Vector{Int}}           # length num_granule_cells
    mf_weights::Matrix{Float64}               # num_granule_cells × num_mossy_fibers_per_granule_cell
    k::Vector{Float64}
    alpha::Vector{Float64}
    bias::Vector{Float64}
    scale::Vector{Float64}
    hyperparameters::Dict{Symbol, AbstractRandomRange}
    
    # Warmup policy
    warmup_multiplier::Float64
    default_warmup_steps::Int
end

function DifferentiatorUnitModel(
        mf,
        hyperparameters::Dict{Symbol, AbstractRandomRange} = model_hyperparameters(DifferentiatorUnitModel);
        num_granule_cells::Int = 100,
        num_mossy_fibers_per_granule_cell::Int = 4,
        warmup_multiplier::Real = 5.0,
        kwargs...)

    model = DifferentiatorUnitModel(
        num_granule_cells,
        num_mossy_fibers_per_granule_cell,
        Vector{Vector{Int}}(),                                                # mf_indices
        zeros(num_granule_cells, num_mossy_fibers_per_granule_cell),          # mf_weights
        zeros(num_granule_cells),                                             # k
        zeros(num_granule_cells),                                             # alpha
        zeros(num_granule_cells),                                             # bias
        ones(num_granule_cells),                                              # scale
        hyperparameters,
        Float64(warmup_multiplier),
        0,
    )
    return init(model, mf, hyperparameters; kwargs...)
end


function init(
        m::DifferentiatorUnitModel,
        mf,
        hyperparameters::Dict{Symbol, AbstractRandomRange};
        rng = Random.default_rng(),
        dt::Real = 1e-3)

    MF = size(mf, 1)
    m.hyperparameters = hyperparameters

    m.mf_indices = [
        rand(rng, 1:MF, m.num_mossy_fibers_per_granule_cell)
        for _ in 1:m.num_granule_cells
    ]
    m.mf_weights = rand(rng, m.num_granule_cells, m.num_mossy_fibers_per_granule_cell)

    longest_tau = 0.0
    for i in 1:m.num_granule_cells
        m.k[i]     = sample(rng, hyperparameters[:k])
        m.alpha[i] = sample(rng, hyperparameters[:alpha])
        m.bias[i]  = sample(rng, hyperparameters[:bias])
        m.scale[i] = 1.0

        k_i = m.k[i]
        k_i > 0 && (longest_tau = max(longest_tau, 1.0 / k_i))
    end

    m.default_warmup_steps = Int(ceil(m.warmup_multiplier * longest_tau / dt))
    return m
end

function Base.show(io::IO, m::DifferentiatorUnitModel)
    print(io, "$(typeof(m)) with $(m.num_granule_cells) granule cells.")
end

function model_hyperparameters(::Type{DifferentiatorUnitModel})
    return Dict{Symbol, AbstractRandomRange}(
        :k     => Uniform(0.0, 50.0),
        :alpha => Uniform(0.0, 0.1),
        :bias  => Uniform(50.0, 100.0),
    )
end

model_hyperparameters(m::DifferentiatorUnitModel) = m.hyperparameters

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::DifferentiatorUnitModel)
    model.hyperparameters[:k] = Uniform(4.62430509057939, 23.585554484690864)
    model.hyperparameters[:alpha] = Uniform(0.01491868890237193, 0.06199395280711233)
    model.hyperparameters[:bias] = Uniform(65.752223139194, 88.85995603078587)
    return model.hyperparameters 
end

function basis(
        m::DifferentiatorUnitModel,
        mf;
        warmup::Integer = 0,
        warmup_mode     = :repeat_first,
        dt::Real        = 1e-3)

    X  = normalize_mf_3d(mf)
    warmup_steps = warmup == 0 ? m.default_warmup_steps : warmup
    Xw = apply_warmup(X; warmup=warmup_steps, warmup_mode=warmup_mode)

    MF, Nt, C = size(Xw)

    length(m.mf_indices) == m.num_granule_cells ||
        error("DifferentiatorUnitModel not initialized. Call init(model, mf_example).")

    B = Array{Float64, 3}(undef, m.num_granule_cells, Nt, C)

    Threads.@threads for i in 1:m.num_granule_cells
        idx = m.mf_indices[i]
        w   = m.mf_weights[i, :]

        k_i     = m.k[i]
        alpha_i = m.alpha[i]
        bias_i  = m.bias[i]
        scale_i = m.scale[i]

        Ac = [alpha_i  1.0; -k_i^2  -2k_i]
        bc = [0.0; k_i^2]
        Ad = I(2) + Ac*dt + (Ac*dt)^2/2 + (Ac*dt)^3/6
        bd = bc * dt

        ImAd     = I(2) - Ad
        use_xinf = abs(det(ImAd)) > 1e-10
        ImAd_inv = use_xinf ? inv(ImAd) : nothing

        for ci in 1:C
            drive = dropdims(sum(Xw[idx, :, ci] .* w, dims=1), dims=1) .- bias_i

            x = use_xinf ? ImAd_inv * (bd * drive[1]) : zeros(2)

            for t in 1:Nt
                x = Ad * x .+ bd .* drive[t]
                B[i, t, ci] = scale_i * max(x[2], 0.0)
            end
        end
    end

    return warmup_steps > 0 ? B[:, warmup_steps+1:end, :] : B
end
