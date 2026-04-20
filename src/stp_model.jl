###############################################################################
# STPDualPoolModel
#
# A rate-based granule cell model with short-term plasticity (STP) implemented
# via a dual transmitter pool synapse.
#
# BIOLOGICAL MOTIVATION
# ---------------------
# This model is inspired by classic short-term synaptic plasticity models
# (Tsodyks & Markram; extended dual-pool variants used in cerebellar MF→GrC
# modeling, e.g. DiGregorio-type formulations).
#
# Each mossy fiber → granule cell synapse contains two transmitter pools:
#   - a FAST pool (readily releasable)
#   - a SLOW pool (reserve)
#
# Transmitter is released from the fast pool in proportion to synaptic
# utilization (u) and presynaptic activity, then replenished from the slow
# pool and through recovery dynamics.
#
# MODELING CHOICES
# ----------------
# - Rate-based (not spike-based): mossy fiber input is treated as an instantaneous
#   firing rate, converted to an expected number of spikes per time step (rate * dt).
# - Facilitation and depression are modeled continuously in time.
# - Granule cells are modeled as leaky integrators of synaptic current with a
#   rectifying (ReLU) output nonlinearity.
# - All parameters are sampled per synapse or per granule cell from specified
#   random ranges (for use in hyperparameter-range optimization).
#
# WARMUP REQUIREMENT
# ------------------
# This model is dynamical and stateful. Synaptic variables (u, R_fast, R_slow)
# and granule cell rates must reach a steady regime before meaningful outputs
# are computed.
#
# Therefore:
#   - Warmup is REQUIRED.
#   - If warmup == 0, the model automatically selects a default warmup duration
#     equal to:
#
#       warmup_multiplier × (longest sampled time constant) / dt
#
#     ensuring all slow processes have time to equilibrate.
#
###############################################################################

using Random
import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange,
    init,
    basis,
    model_hyperparameters

import ..Uniform, ..sample

export STPDualPoolModel


###############################################################################
# Dual-pool STP synapse
###############################################################################

"""
TwoPoolSynapse

Represents a single MF→GrC synapse with:
- facilitation (u)
- a fast transmitter pool
- a slow transmitter pool

State variables evolve continuously and are driven by presynaptic firing rate.
"""
mutable struct TwoPoolSynapse
    # -------------------------
    # Parameters (sampled)
    # -------------------------
    w0::Float64        # synaptic weight / gain
    U::Float64         # baseline utilization
    tau_u::Float64     # facilitation time constant
    tau_fast::Float64  # fast pool replenishment time constant
    tau_slow::Float64  # slow pool recovery time constant

    # -------------------------
    # State variables
    # -------------------------
    u::Float64         # current utilization
    R_fast::Float64    # fast pool availability
    R_slow::Float64    # slow pool availability
end

"""
Constructor initializes synapse at a biologically reasonable baseline state.
"""
function TwoPoolSynapse(; w0, U, tau_u, tau_fast, tau_slow)
    TwoPoolSynapse(
        w0, U, tau_u, tau_fast, tau_slow,
        U,        # u starts at baseline
        0.20,     # fraction of transmitter in fast pool
        1.00      # slow pool fully available
    )
end

"""
Reset synapse state (used before each condition / warmup).
"""
function reset!(s::TwoPoolSynapse)
    s.u = s.U
    s.R_fast = 0.20
    s.R_slow = 1.00
    return nothing
end

"""
step!(s, rate, dt) -> synaptic current

- `rate` is a presynaptic firing rate (Hz-like).
- Converted internally to expected spike count = rate * dt.
"""
function step!(s::TwoPoolSynapse, rate::Float64, dt::Float64)
    spikes = rate * dt # Probability of a spike per timestep

    # Facilitation dynamics
    s.u += dt * (s.U - s.u) / s.tau_u
    s.u += s.U * (1 - s.u) * spikes
    s.u = clamp(s.u, 0.0, 1.0)

    # Release from fast pool
    # Note: can't release more than we actually have in the fast pool
    release = min(s.u * s.R_fast * spikes, s.R_fast)
    s.R_fast -= release

    # Transfer from slow → fast pool
    Δ = dt * (s.R_slow - s.R_fast) / s.tau_fast
    s.R_fast += Δ
    s.R_slow -= Δ

    # Slow pool recovery
    s.R_slow += dt * (1.0 - s.R_slow) / s.tau_slow

    # Numerical safety
    s.R_fast = clamp(s.R_fast, 0.0, 1.0)
    s.R_slow = clamp(s.R_slow, 0.0, 1.0)

    # We divide by dt to turn it back into units of firing rate
    # rather than probability per timestep
    return s.w0 * release / dt
end


###############################################################################
# Granule cell unit
###############################################################################

"""
STPGranuleCell

A single granule cell:
- integrates synaptic current from multiple STP synapses
- evolves a firing rate via leaky integration
- applies a rectifying output nonlinearity
"""
mutable struct STPGranuleCell
    tau_m::Float64                       # membrane time constant
    bias::Float64                       # subtractive bias / threshold
    synapses::Vector{TwoPoolSynapse}    # MF→GrC synapses
    rate::Float64                       # current firing rate
end

"""
Reset granule cell and all synapses.
"""
function reset!(gc::STPGranuleCell)
    gc.rate = 0.0
    for s in gc.synapses
        reset!(s)
    end
    return nothing
end

"""
Advance granule cell dynamics by one time step.
"""
function step!(gc::STPGranuleCell, inputs::Vector{Float64}, dt::Float64)
    I_syn = 0.0
    @inbounds for j in 1:length(gc.synapses)
        I_syn += step!(gc.synapses[j], inputs[j], dt)
    end

    # Leaky rate dynamics
    d_rate = (-gc.rate + I_syn - gc.bias) / gc.tau_m
    gc.rate += dt * d_rate
    
    # Convert to firing rates (divide by dt) 
    # and subtract off bias, apply RELU
    firing_rate = max(gc.rate, 0.0) # RELU
    return firing_rate
end


###############################################################################
# STP Dual-Pool Model
###############################################################################

"""
STPDualPoolModel

Population of granule cells with:
- random MF connectivity
- per-synapse STP dynamics
- per-cell membrane dynamics

Designed to be used as a basis-set generator.
"""
mutable struct STPDualPoolModel <: AbstractGRCBasisModel
    num_granule_cells::Int
    num_mossy_fibers_per_granule_cell::Int

    mf_indices::Vector{Vector{Int}}       # connectivity
    granule_cells::Vector{STPGranuleCell}

    hyperparameters::Dict{Symbol, AbstractRandomRange}

    # Warmup policy
    warmup_multiplier::Float64
    default_warmup_steps::Int
end

function STPDualPoolModel(mf,
        hyperparameters::Dict{Symbol, AbstractRandomRange}=model_hyperparameters(STPDualPoolModel);
        num_granule_cells::Int=100,
        num_mossy_fibers_per_granule_cell::Int=4,
        warmup_multiplier::Real=5.0,
        kwargs...)

    model = STPDualPoolModel(
        num_granule_cells,
        num_mossy_fibers_per_granule_cell,
        Vector{Vector{Int}}(),
        STPGranuleCell[],
        hyperparameters,
        Float64(warmup_multiplier),
        0,
    )

    init(model, mf, hyperparameters; kwargs...)
end

function Base.show(io::IO, model::STPDualPoolModel)
    print(io, "$(typeof(model)) with $(model.num_granule_cells) granule cells.")
end

"""
Hyperparameter ranges (optimized externally).
"""
function model_hyperparameters(::Type{STPDualPoolModel})
    Dict{Symbol,AbstractRandomRange}(
        :U        => Uniform(0.0, 1.0),
        :tau_fast => Uniform(1e-3, 500e-3),
        :tau_slow => Uniform(10e-3, 5000e-3),
        :bias => Uniform(0, 3) # 3
    )
    # At the moment, don't optimize w0, we use the same 
    # as for every other model (0-1) random weights
    #:w0       => Uniform(100.0, 500.0)
end

model_hyperparameters(m::STPDualPoolModel) = m.hyperparameters

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::STPDualPoolModel)
    #model.hyperparameters[:U] = Uniform(0.00018886152821313382, 0.11443441635517)
    #model.hyperparameters[:bias] = Uniform(0.00039372878874559177, 0.00042973103256)
    #model.hyperparameters[:tau_fast] = Uniform(0.009882063395088216, 0.0804781059391187)
    #model.hyperparameters[:tau_slow] = Uniform(3.838769908315781, 4.9967582395715695)
    
    model.hyperparameters[:U] = Uniform(0.11136412856280867, 0.2604631251900723)
    model.hyperparameters[:bias] = Uniform(0.03570009700841181, 1.316643340643353)
    model.hyperparameters[:tau_fast] = Uniform(0.1308221078549704, 0.2379549852334993)
    model.hyperparameters[:tau_slow] = Uniform(0.548905012120658, 3.4773749012618174)
    return model.hyperparameters
end

###############################################################################
# Initialization
###############################################################################

function init(m::STPDualPoolModel, mf, hyperparameters; rng=Random.default_rng(), dt=1e-3)
    MF = size(mf, 1)
    m.hyperparameters = hyperparameters

    # Random MF connectivity
    m.mf_indices = [
        rand(rng, 1:MF, m.num_mossy_fibers_per_granule_cell)
        for _ in 1:m.num_granule_cells
    ]

    m.granule_cells = Vector{STPGranuleCell}(undef, m.num_granule_cells)
    longest_tau = 0.0

    for i in 1:m.num_granule_cells
        tau_m = sample(rng, Uniform(1e-3, 5e-3)) # We don't optimize this parameter, not a free parameter
        bias  = sample(rng, hyperparameters[:bias])

        syns = Vector{TwoPoolSynapse}(undef, m.num_mossy_fibers_per_granule_cell)
        for j in 1:m.num_mossy_fibers_per_granule_cell
            syns[j] = TwoPoolSynapse(
                w0       = sample(rng, Uniform(0, 1)),
                U        = sample(rng, hyperparameters[:U]),
                tau_u    = sample(rng, Uniform(2e-3, 10e-3)), # Limited to no potentiation in this model, not a free parameter
                tau_fast = sample(rng, hyperparameters[:tau_fast]),
                tau_slow = sample(rng, hyperparameters[:tau_slow])
            )
            longest_tau = max(longest_tau, syns[j].tau_u, syns[j].tau_fast, syns[j].tau_slow)
        end

        longest_tau = max(longest_tau, tau_m)
        m.granule_cells[i] = STPGranuleCell(tau_m, bias, syns, 0.0)
    end

    # Automatic warmup duration
    m.default_warmup_steps = Int(ceil(m.warmup_multiplier * longest_tau / dt))
    return m
end


###############################################################################
# Basis computation
###############################################################################

function basis(m::STPDualPoolModel, mf; warmup=0, dt=1e-3)
    X = normalize_mf_3d(mf)
    MF, T, C = size(X)

    warmup_steps = warmup == 0 ? m.default_warmup_steps : warmup
    warmup_steps > 0 || error("STPDualPoolModel requires warmup > 0")

    B = Array{Float64,3}(undef, m.num_granule_cells, T, C)
    inputs = Vector{Float64}(undef, m.num_mossy_fibers_per_granule_cell)

    for i in 1:m.num_granule_cells
        I  = m.mf_indices[i]
        gc = m.granule_cells[i]

        for c in 1:C
            reset!(gc)

            # Warmup with constant input
            for j in 1:length(I)
                inputs[j] = X[I[j], 1, c]
            end
            for _ in 1:warmup_steps
                step!(gc, inputs, dt)
            end

            # Full simulation
            for t in 1:T
                for j in 1:length(I)
                    inputs[j] = X[I[j], t, c]
                end
                B[i, t, c] = step!(gc, inputs, dt)
            end
        end
    end

    return B
end
