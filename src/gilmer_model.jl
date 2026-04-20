using Random
import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    AbstractRandomRange,
    init,
    basis,
    model_hyperparameters

import ..Uniform, ..sample

export ThresholdedGilmerModel

"""
Thresholded Gilmer granule-cell model.

Each granule cell:
1) selects a fixed subset of mossy fibers
2) forms a 1D drive over time (weighted sum or mean)
3) computes a threshold = mean(drive) + z * std(drive)
4) outputs ReLU(drive - threshold)

Notes:
- z is sampled per-granule-cell from hyperparameter range :z
- weights are sampled per-granule-cell if :use_weights is true
- threshold statistics are computed on the post-warmup segment to avoid warmup bias
"""
mutable struct ThresholdedGilmerModel <: AbstractGRCBasisModel
    num_granule_cells::Int
    num_mossy_fibers_per_granule_cell::Int

    mf_indices::Vector{Vector{Int}}                      # length num_granule_cells
    mf_weights::Matrix{Float64}                          # num_granule_cells × num_mossy_fibers_per_granule_cell
    z_per_granule_cell::Vector{Float64}                  # length num_granule_cells
    mean_input::Vector{Float64}
    std_input::Vector{Float64}

    use_weights::Bool
    normalize_weights::Bool
    tau_m::Vector{Float64}                       # membrane time constant
    hyperparameters::Dict{Symbol, AbstractRandomRange}
end

"""
    Gilmer/Person Granule Basis

Use the model of Gilmer et al. (JNP, 2023) to predict the responses of Purkinje cells. The model archiecture is straight-forward:
```math
GRC_i(t) = \\left( \\left[ \\sum_{i}^{4}  \\frac{MF_{k}(t)}{4} \\right] - \\theta_i \\right)_+
```

The parameter ``\theta``, corresponding to the bias equal to
```math
\\theta_i = \\bar{MF} + z \\sigma(MF)
```
The parameter ``\\bar{MF}`` corresponds to the mean across connected mossy fibers (best I can tell from the
 paper, not across the complete mossy fiber population). The parameter ``\\sigma(MF)`` 
 corresponds to the standard deviation of the weighted mossy fiber 
 inputs in time. The only free parameter is `z``, essentially a 
 threshold on the z-scored input.
"""
function ThresholdedGilmerModel(mf,
        hyperparameters::Dict{Symbol, AbstractRandomRange}=model_hyperparameters(ThresholdedGilmerModel);
        num_granule_cells::Int=100,
        num_mossy_fibers_per_granule_cell::Int=4,
        use_weights::Bool=false,
        normalize_weights::Bool=true,
        kwargs...)
    
    model = ThresholdedGilmerModel(
        num_granule_cells,
        num_mossy_fibers_per_granule_cell,
        Vector{Vector{Int}}(),
        zeros(num_granule_cells, num_mossy_fibers_per_granule_cell),
        zeros(num_granule_cells),
        zeros(num_granule_cells),
        zeros(num_granule_cells),
        use_weights,
        normalize_weights,
        zeros(num_granule_cells),
        hyperparameters
    )
    init(model, mf, hyperparameters; kwargs...)
end

function init(m::ThresholdedGilmerModel, mf, hyperparameters::Dict{Symbol, AbstractRandomRange}; rng=Random.default_rng())
    MF = size(mf, 1)

    m.mf_indices = [
        rand(rng, 1:MF, m.num_mossy_fibers_per_granule_cell)
        for _ in 1:m.num_granule_cells
    ]
    

    m.hyperparameters = hyperparameters

    # z sampled per GRC from range
    for i in 1:m.num_granule_cells
        m.z_per_granule_cell[i] = sample(rng, m.hyperparameters[:z])
    end

    # weights: either all-ones (mean) or random positive weights
    if m.use_weights
        m.mf_weights .= rand(rng, m.num_granule_cells, m.num_mossy_fibers_per_granule_cell)

        if m.normalize_weights
            # normalize each row to sum to 1 (so drive is a weighted mean)
            for i in 1:m.num_granule_cells
                s = sum(m.mf_weights[i, :])
                s > 0 || error("ThresholdedGilmerModel: sampled zero-sum weights unexpectedly.")
                m.mf_weights[i, :] ./= s
            end
        end
    else
        # Equal weights -> simple mean if normalized, or sum if not
        m.mf_weights .= 1.0
        if m.normalize_weights
            m.mf_weights ./= m.num_mossy_fibers_per_granule_cell
        end
    end

    # Compute the mean and standard deviation of all weighted granule cell inputs
    X  = normalize_mf_3d(mf) 
    for i in 1:m.num_granule_cells
        I = m.mf_indices[i]
        w = m.mf_weights[i, :]
        drive = dropdims(sum(X[I, :, :] .* w, dims=1), dims=1)
        m.mean_input[i] = mean(drive)
        m.std_input[i] = std(drive)
        m.tau_m[i] = sample(rng, Uniform(1e-3, 5e-3)) # We don't optimize this parameter, not a free parameter
    end

    return m
end

function Base.show(io::IO, model::ThresholdedGilmerModel)
    print(io, "$(typeof(model)) with $(model.num_granule_cells) granule cells.")
end

function model_hyperparameters(::Type{ThresholdedGilmerModel})
    # z controls sparsity via threshold = mean + z * std
    # Positive z => higher threshold => sparser outputs
    return Dict{Symbol,AbstractRandomRange}(
        :z => Uniform(0, 10.0),
    )
end

function model_hyperparameters(model::ThresholdedGilmerModel)
    return model.hyperparameters
end

"""
    return the optimal hyperparameters (after we optimized)

If the parameters of the model change, you will have to re-optimized, but
these are the ideal parameters for the current model, allowing us to instantiate
previously optimized models.
"""
function optimal_hyperparameters!(model::ThresholdedGilmerModel)
    model.hyperparameters[:z] = Uniform(1.5924992885505709, 1.6651669638150015)
    #model.hyperparameters[:z] = Uniform(0.18792524156822796, 0.2113264150977331)
    return model.hyperparameters
end

function basis(m::ThresholdedGilmerModel, mf; warmup::Integer=0, warmup_mode=:repeat_first, dt::Real=1e-3)
    X  = normalize_mf_3d(mf)                                   # MF × T × C
    Xw = apply_warmup(X; warmup=warmup, warmup_mode=warmup_mode)

    MF, Nt, C = size(Xw)

    length(m.mf_indices) == m.num_granule_cells ||
        error("ThresholdedGilmerModel not initialized. Call init(model, mf_example).")

    B = Array{Float64,3}(undef, m.num_granule_cells, Nt, C)

    # segment used for threshold statistics (exclude warmup padding)
    stats_start = warmup > 0 ? warmup + 1 : 1

    for i in 1:m.num_granule_cells
        I = m.mf_indices[i]
        w = m.mf_weights[i, :]
        z = m.z_per_granule_cell[i]

        for c in 1:C
            # drive(t) = sum_j Xw[I[j], t, c] * w[j]
            drive = dropdims(sum(Xw[I, :, c] .* w, dims=1), dims=1)  # length Nt
            thr = m.mean_input[i] + z * m.std_input[i]

            # Leaky rate dynamics
            rate = max(0.0, drive[1] - thr)
            for t = 1:Nt
                d_rate = (-rate + drive[t] - thr) / m.tau_m[i]
                rate += dt * d_rate
                rate = max(rate, 0.0) # RELU
                B[i, t, c] = rate
            end

            # thresholded output
            #@. B[i, :, c] = max(0.0, drive - thr)
        end
    end

    return warmup > 0 ? B[:, warmup+1:end, :] : B
end
