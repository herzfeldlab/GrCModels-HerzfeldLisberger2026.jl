using Random
import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange

import ..Uniform, ..sample

export RandomProjectionModel

"""
Random Projection Model by Yamazaki & Tanaka

Each granule cell:

Notes:

"""
mutable struct RandomProjectionModel <: AbstractGRCBasisModel
    num_granule_cells::Int
    num_mossy_fibers_per_granule_cell::Int

    mf_indices::Vector{Vector{Int}} # length num_granule_cells
    mf_weights::Matrix{Float64}  # num_granule_cells × num_mossy_fibers_per_granule_cell
    recurrent_weights::Matrix{Float64} # num_granule_cells x num_granule_cells
    tau::Vector{Float64}  # synaptic integration time constant, length num_granule cells
    k::Vector{Float64} # Synapse connection probability
    tau_m::Vector{Float64} # membrane time constant
    bias::Vector{Float64}
    hyperparameters::Dict{Symbol, AbstractRandomRange}
    # Warmup policy
    warmup_multiplier::Float64
    default_warmup_steps::Int
end

"""
 # 'Random Projection" Model by Yamazaki & Tanaka

A model proposed by Yamazaki & Tanaka (Neural Comput., 2005) based on initial work 
by Buonomano & Mauk (1994) and Medina, ... Mauke (J Neurosci, 2000). 
The basic idea is that granule cells undergo random 
transitions between active and inactive states. Thus a complete population
of granule cells encode a unique time after conditioned stimulus onset. Yamazaki & 
Tanka's model is given by the following differential equation:
```math
    GRC_i(t+1) = \\left[ I_i(t) -  \\sum_{s=0}^t \\exp \\left(- \\frac{t - s}{\\tau} \\right) \\sum_{j=1}^{N} w_{ij} GRC_j(s) \\right]_+
```

The activity of the granule cell is half-wave rectified and 
is a function of its inputs, I(i), as well as the time weighted (by an exponential decay) 
input from other neurons via the connection matrix, ``w_{ij}``. The connection 
matrix was set based on a binomial distribution 
with ``Pr(w_{ij}) = 0 = Pr( W_{ij}=2\\kappa/N ) = 0.5``.
Here, `N`` is the number of units. Thus, the model contains three free parameters:
kappa, tau and the bias.
"""
function RandomProjectionModel(mf,
        hyperparameters::Dict{Symbol, AbstractRandomRange}=model_hyperparameters(RandomProjectionModel);
        num_granule_cells::Int=100,
        num_mossy_fibers_per_granule_cell::Int=4,
        warmup_multiplier::Real=5.0, kwargs...)
    
    model = RandomProjectionModel(
        num_granule_cells,
        num_mossy_fibers_per_granule_cell,
        Vector{Vector{Int}}(),
        zeros(num_granule_cells, num_mossy_fibers_per_granule_cell),
        zeros(num_granule_cells, num_granule_cells),
        zeros(num_granule_cells),
        zeros(num_granule_cells),
        zeros(num_granule_cells),
        zeros(num_granule_cells),
        hyperparameters,
        Float64(warmup_multiplier),
        0.0
    )
    init(model, mf, hyperparameters; kwargs...)
end

function init(m::RandomProjectionModel, mf, hyperparameters::Dict{Symbol, AbstractRandomRange}; rng=Random.default_rng(), dt=1e-3)
    MF = size(mf, 1)

    m.mf_indices = [
        Random.randperm(rng, MF)[1:m.num_mossy_fibers_per_granule_cell]
        for _ in 1:m.num_granule_cells
    ]
    
    # Set our weights randomly
    m.mf_weights .= rand(rng, m.num_granule_cells, m.num_mossy_fibers_per_granule_cell)

    m.hyperparameters = hyperparameters

    # Compute the mean and standard deviation of all weighted granule cell inputs
    longest_tau = 0
    for i in 1:m.num_granule_cells
        m.tau_m[i] = sample(rng, Uniform(1e-3, 5e-3)) # We don't optimize this parameter, not a free parameter
        m.tau[i] = sample(rng, hyperparameters[:tau])
        m.k[i] = sample(rng, hyperparameters[:k])
        m.bias[i] = sample(rng, hyperparameters[:bias])

        # Now that we have sampled k, we can define our recurrent weight matrix
        # Pr(wij = 0) = Pr(wij = 2κ/N) = 0.5
        m.recurrent_weights[i, :] = rand([0.0, 2 * m.k[i] / m.num_granule_cells], m.num_granule_cells)
        longest_tau = max(longest_tau, m.tau_m[i], m.tau[i])
    end

    # Automatic warmup duration
    m.default_warmup_steps = Int(ceil(m.warmup_multiplier * longest_tau / dt))

    return m
end

function Base.show(io::IO, model::RandomProjectionModel)
    print(io, "$(typeof(model)) with $(model.num_granule_cells) granule cells.")
end

function model_hyperparameters(::Type{RandomProjectionModel})
    # k controls the distribution of weights for initialization
    # tau controls the input time constant (synaptic time constant)
    return Dict{Symbol,AbstractRandomRange}(
        :k => Uniform(0, 1.0),
        :tau => Uniform(1e-3, 250e-3),
        :bias => Uniform(0, 150)
    )
end

function model_hyperparameters(model::RandomProjectionModel)
    return model.hyperparameters
end

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::RandomProjectionModel)
    model.hyperparameters[:k] = Uniform(0.2966132873070939, 0.962643904127803)
    model.hyperparameters[:tau] = Uniform(0.13321105192567054, 0.1788903335687331)
    model.hyperparameters[:bias] = Uniform(1.2913354816352698, 1.9513343420670446)
    
    #model.hyperparameters[:k] = Uniform(0.8644978740640545, 0.9719310194284246)
    #model.hyperparameters[:tau] = Uniform(0.004924043985823328, 0.09928726100267639)
    #model.hyperparameters[:bias] = Uniform(13.674995074662624, 14.234819627863818)
    return model.hyperparameters
end

function basis(m::RandomProjectionModel, mf; warmup::Integer=0, warmup_mode=:repeat_first, dt::Real=1e-3)
    X  = normalize_mf_3d(mf)               
    
    warmup_steps = warmup == 0 ? m.default_warmup_steps : warmup
    warmup_steps > 0 || error("RandomProjectionModel requires warmup > 0")
    warmup = warmup_steps

    # MF × T × C
    Xw = apply_warmup(X; warmup=warmup, warmup_mode=warmup_mode)

    MF, Nt, C = size(Xw)

    length(m.mf_indices) == m.num_granule_cells ||
        error("RandomProjectionMode not initialized. Call init(model, mf_example).")

    B = Array{Float64,3}(undef, m.num_granule_cells, Nt, C)

     for c in 1:C
        last_rate_resp = zeros(m.num_granule_cells)
        last_recurrent_drive = zeros(m.num_granule_cells)
        for t = 1:Nt
            for i = 1:m.num_granule_cells
                I = m.mf_indices[i]
                w = m.mf_weights[i, :]

                mf_drive = sum(Xw[I, t, c] .* w)
                recurrent_drive = sum(m.recurrent_weights[i, :] .* last_rate_resp)

                # Filter current drive using the neuron's time constant
                filtered_recurrent_drive = last_recurrent_drive[i]
                d_filtered_recurrent_drive = (-filtered_recurrent_drive + recurrent_drive) / m.tau[i]
                filtered_recurrent_drive += dt * d_filtered_recurrent_drive
                last_recurrent_drive[i] = filtered_recurrent_drive
                
                # Leaky rate dynamics
                rate = last_rate_resp[i]
                d_rate = (-rate + mf_drive - filtered_recurrent_drive) / m.tau_m[i]
                rate += dt * d_rate
                rate = rate .- m.bias[i] # Remove bias
                rate = max(rate, 0.0) # RELU
                B[i, t, c] = rate
            end
            last_rate_resp = B[:, t, c]
        end
    end

    return warmup > 0 ? B[:, warmup+1:end, :] : B
end
