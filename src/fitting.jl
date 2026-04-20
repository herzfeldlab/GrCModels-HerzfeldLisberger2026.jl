import ..GrCModelsInterface:
    AbstractGRCBasisModel,
    init,
    basis,
    model_hyperparameters

using NonNegLeastSquares: nonneg_lsq
import LinearAlgebra
import Optim
import BlackBoxOptim

export fit_purkinje_cell_responses,
       optimize_hyperparameters

"""
    fit_purkinje_cell_responses(model, mf, pc)

Construct a linear fit between granule cell basis sets and Purkinje Cell
responses. By definition, weights between mossy fibers and Purkinje Cells
must be strictly positive. Thus we use one of two potential approaches to determine
the weights. When method is :gradient_descent, we use non-negative least squares
to to determine the weights. When the method is :pf_pc_learning we use an LTP/LTD
learning rule to determine the weights.

This function removes the baseline period from Purkinje cell (output responses)
as well as from the additional regressors. This ensures that the granule cell
responses fit only the change in output rather than the baseline period. This
has the benefit of enforcing sparsity at the level of the granule cell basis (because
baseline responses must remain close to zero).

Arguments:
 - `model`: The basis set model to use to determine granule cell responses given `mf`
 - `mf`: Mossy fiber responses arranged as num mossy fibers x num timepoints x num conditions.
 - `pc`: Purkinje cell responses arranged as num purkinje cells x num timepoints x num conditions
 - `additional_regressors`: Additional regressors used to fit Purkinje cell responses (probably MLIs)
 - `baseline_period`: A tuple (or nothing) describing the start and stop indices to use to subtract off
     baseline responses.
 - `method`: One of :gradient_descent or :pf_pc_learning, which determines 
    the method used for fitting.

Returns:
 - The fit purkinje cell responses
"""
function fit_purkinje_cell_responses(
    model::AbstractGRCBasisModel,
    mf,
    pc;
    bias::Bool = false,
    additional_regressors = nothing,
    remove_reconstruction_bias::Bool = false,
    baseline_period::Union{Nothing, Tuple{<:Integer, <:Integer}} = (1, 100),
    method::Union{Nothing, Symbol}=nothing,
    kwargs...
)

    if method === nothing && haskey(ENV, "pc_fit_method")
        method = Symbol(ENV["pc_fit_method"])
    end

    # --- Normalize inputs ---
    mf = normalize_mf_3d(mf)
    pc = normalize_mf_3d(pc)

    (size(mf,2) == size(pc,2) && size(mf,3) == size(pc,3)) ||
        error("MF and PC dimensions must match (time, conditions)")

    if additional_regressors !== nothing
        (size(mf,2) == size(additional_regressors,2) &&
         size(mf,3) == size(additional_regressors,3)) ||
            error("Additional regressors must match MF dimensions")
    end

    # --- Construct granule cell basis ---
    X = basis(model, mf)

    # --- Optional baseline subtraction (PC only) ---
    b_pc = zeros(size(pc,1))
    if baseline_period !== nothing
        b_pc = dropdims(mean(pc[:, baseline_period[1]:baseline_period[2], :], dims=(2,3)), dims=(2,3))
        pc = pc .- b_pc
    end

    # --- Flatten across conditions ---
    X_flat  = reshape(X, size(X,1), :)
    Y_flat  = reshape(pc, size(pc,1), :)

    # ============================================================
    # Add constant offset regressor (bias)
    # ============================================================
    T_total = size(X_flat, 2)
    X_aug = copy(X_flat)

    # New addition
    if additional_regressors !== nothing
        additional_regressors_flat = reshape(additional_regressors, size(additional_regressors, 1), :)
        X_aug = vcat(X_flat, additional_regressors_flat)
    end

    # ============================================================
    # Fit PF weights + bias on residuals
    # ============================================================
    if method === nothing || method == :cerebellar_learning_rule
        W_pf, Y_hat_flat, b = pf_pc_learning(X_aug, Y_flat; bias=bias, kwargs...)
    elseif method == :gradient_descent
        W_pf, Y_hat_flat = nn_lsq(X_aug, Y_flat)
        b = zeros(size(Y_flat, 1))
    else
        error("Invalid method for determine granule cell weights")
    end

    # Reshape back to original dimensions
    Y_hat = reshape(Y_hat_flat, size(pc))

    if remove_reconstruction_bias == true
        recon_baseline = mean(Y_hat[:, baseline_period[1]:baseline_period[2], :], dims=(2, 3))
        Y_hat .-= recon_baseline
    end

    # Add baseline back
    if baseline_period !== nothing
        Y_hat .+= b_pc
    end

    # Return the weights and the reconstrution
    if additional_regressors === nothing
        return W_pf, Y_hat
    end
    W_additional = W_pf[:, (size(X_flat, 1 )+1):end]
    additional_input = W_additional * additional_regressors_flat
    #return W_pf, Y_hat, B * A
    return W_pf[:, 1:size(X_flat, 1)], Y_hat, additional_input
end

"""
    nn_lsq(X, Y, max_iter=100; dropout::Real=0.25, num_replicates::Integer=25)

Nonnegative least squares (NNLS) mapping from granule activity `X` (N×T) to Purkinje activity `Y` (M×T),
with *dropout-induced weight spreading* controlled by `dropout`.

Why this works (math):

Let `G = X'` be the usual design matrix (T×N), `y` a single Purkinje trace (T,). Standard NNLS solves:

    min_{w ≥ 0}  || y - G w ||².

To make weights robust to missing / unreliable regressors (feature dropout), assume a random Bernoulli mask
`m ∈ {0,1}^N` where each feature is kept with probability `p = 1 - dropout`.

Use *inverted dropout scaling* so the predictor is unbiased in expectation:

    ŷ(m) = G ((m ⊙ w)/p).

Then define weights by minimizing expected squared error under this masking:

    min_{w ≥ 0}  E_m [ || y - G ((m ⊙ w)/p) ||² ].

A standard calculation gives the closed form:

    E_m [ || y - G ((m ⊙ w)/p) ||² ]
      = || y - G w ||²  +  ((1-p)/p) * Σ_i ||G[:,i]||² * w_i².

So dropout induces an ℓ2-type penalty with *fixed* strength α = (1-p)/p and per-feature scaling ||G[:,i]||².
This encourages distributing mass across (near-)colinear predictors because concentrating weight on any one feature
is expensive (it increases the expected loss when that feature is dropped).

We can solve this *exactly* as an augmented NNLS:

    min_{w ≥ 0} || [y; 0] - [G; sqrt(α)*D] w ||²

where D = diag(||G[:,1]||, …, ||G[:,N]||).

Notes:
- `max_iter` is the number of nonnegative least squares iterations
- `dropout=0` reduces to standard NNLS.
- `dropout→1` makes α blow up; we clamp to avoid division by zero.
"""
function nn_lsq(X, Y; max_iter::Integer=100, dropout::Real=0.0)
    N, T  = size(X)     # N predictors (granule cells), T samples
    M, Ty = size(Y)     # M outputs (Purkinje cells), Ty samples
    @assert T == Ty "X and Y must have the same number of samples"
    # Dropout fraction q in [0,1); keep prob p = 1-q
    q = clamp(float(dropout), 0.0, 1.0)
    # Avoid p=0 (q=1) which would imply infinite α
    p = max(1.0 - q, eps(Float64))
    α = (1.0 - p) / p    # = q/(1-q)

    # Design matrix for NNLS: G is T×N (each column is one granule cell time series)
    G = X'  # T×N

    # Per-feature scale in the dropout-induced penalty: ||G[:,i]||
    # (equivalently, ||X[i,:]|| since column i of G is row i of X)
    d = vec(norm.(eachcol(G)))     # length N
    D = Diagonal(d)                # N×N

    # Augmented system encodes: ||y-Gw||² + α*||D w||²
    # with nonnegativity constraint w ≥ 0.
    G_aug = (α == 0.0) ? G : vcat(G, sqrt(α) * D)  # (T+N)×N
    W = zeros(M, N)

    for m in 1:M
        y = @view Y[m, :]                 # length T
        y_aug = (α == 0.0) ? collect(y) : vcat(collect(y), zeros(N))
        W[m, :] = nonneg_lsq(G_aug, y_aug, alg=:fnnls, max_iter=max_iter)
    end

    Y_hat = W * X
    return W, Y_hat
end


function pf_pc_learning(X, Y;
    max_iter::Integer = 1_000,
    bias::Bool        = false,
    w_tol::Real       = 1e-6,
    y_tol::Real       = 1e-6,
    progress::Bool    = false)

    N, T = size(X)
    M, _ = size(Y)

    raw_rms = sqrt.(mean(abs2.(X), dims=2)) |> vec
    ε       = 1e-6 
    scale   = raw_rms .+ ε
    X_s     = X ./ scale

    # Calculate the average power per granule cell
    avg_power = sum(abs2, X_s) / (N * T)

    # Base rate scaled by population size and power
    # This ensures the total 'push' on the Purkinje cell stays constant regardless of N
    base_rate = 0.1 / (avg_power * N)

    # Alpha (LTP) is additive; Eta (LTD) is multiplicative.
    alpha = base_rate 
    eta   = base_rate / 10.0 

    W = zeros(M, N) # rand(M, N) ./ N           # M × N
    err_prev = fill(Inf, M, T)
    b = zeros(M) 

    p = progress ? ProgressMeter.ProgressUnknown("Fitting") : nothing

    # Allow negative 

    for iter in 1:max_iter
        W_prev = copy(W)
        Y_hat = (W * X_s)
        if bias
            b = mean(Y, dims=2) .- mean(Y_hat, dims=2)
        end

        Y_hat = Y_hat .+ b 
        err = Y_hat .- Y # M × T
             
        # This original rule requires that regressors were strictly positive
        # The revised code allows regressors to be negative (with positive
        # weights).
        #err_pos = max.(err,  0.0)
        #err_neg = max.(-err, 0.0)
        #ltp = (alpha / T) .* (err_neg * X_abs')   # flip direction for inhibitory units
        #ltd = (eta / T ) .* W .* (err_pos * X_abs') # always suppressive, no sign flip needed
        #W .+= (ltp .- ltd)
        
        # Revise code is below, which accounts for negative regressors and appropriately
        # uses LTP/LTD for negative and positive regressors.

        # Compute the correlation (gradient) for each weight
        # G[m, n] > 0 means the weight needs to decrease (LTD)
        # G[m, n] < 0 means the weight needs to increase (LTP)
        G = (err * X_s') ./ T
        ltd_mask = max.(G, 0.0)
        ltp_mask = min.(G, 0.0)

        # Apply Multiplicative LTD and Additive LTP
        # dW = - (eta * W * ltd_grad) - (alpha * ltp_grad)
        # We subtract because ltp_mask is already negative.
        W .-= (eta .* W .* ltd_mask)  # Multiplicative
        W .-= (alpha .* ltp_mask)     # Additive
        W  .= max.(W, 0.0)

        Δw   = norm(W .- W_prev)   / (norm(W_prev)   + eps())
        Δerr = norm(err .- err_prev) / (norm(err_prev) + eps())
        err_prev = copy(err)
        progress && ProgressMeter.next!(p, showvalues=[("Error", norm(err))])

        (Δw < w_tol && Δerr < y_tol) && break
        
    end
    progress && ProgressMeter.finish!(p)
    W     = W ./ scale'
    Y_hat = (W * X) .+ b
    return W, Y_hat, b
end

"""
    optimize_hyperparameters(model, mf, pc; num_iterations=100, verbose=false, kwargs...)

Optimize the search ranges of a model’s hyperparameters using black-box optimization.

This function adjusts the `(low, high)` bounds associated with each hyperparameter
of `model` in order to minimize prediction error on observed data. The optimization
is performed over *ranges* rather than fixed parameter values, allowing the model
to subsequently sample within improved bounds.

# Overview
- Each hyperparameter is represented by a pair `(low, width)` encoded in a normalized
  parameter vector `θ ∈ [0,1]^(2K)`, where `K` is the number of hyperparameters.
- These normalized values are mapped back to concrete `(low, high)` ranges using
  fixed, declared bounds (which are never mutated).
- A black-box optimizer (`BlackBoxOptim.jl`) searches over this space to minimize
  mean squared error (MSE) between model predictions and observed outputs.

# Objective Function
For each candidate `θ`:
- Convert normalized parameters into valid `(low, high)` ranges.
- Reinitialize the model with these ranges.
- Fit the model multiple times (with different RNG seeds) to reduce stochastic variance.
- Compute the average MSE between predicted (`ŷ`) and observed (`pc`) responses.
- Return `Inf` if fitting fails due to numerical issues (e.g., singular matrices).

# Arguments
- `model`: Model instance whose hyperparameter ranges will be optimized.
- `mf`: Input features (e.g., mossy fiber activity), expected to be 3D (neuron x time x condition).
- `pc`: Output responses (e.g., Purkinje cell activity), also expected to be 3D.

# Keyword Arguments
- `num_iterations::Integer=100`: Maximum number of optimization steps.
- `verbose::Bool=false`: If `true`, prints intermediate optimization progress.
- `kwargs...`: Additional keyword arguments passed to `BlackBoxOptim.bboptimize`.

# Returns
- `hyperparameters`: A dictionary of hyperparameters with updated `(low, high)` bounds.

# Notes
- The original declared bounds of each hyperparameter define a fixed search space
  and are not modified during optimization.
- Each hyperparameter range is constrained to have a minimum width (`1e-6`) to
  avoid degenerate intervals.
- The optimization is stochastic and may yield different results across runs.
- Multiple replicates per evaluation improve robustness to random initialization.
"""
function optimize_hyperparameters(model, mf, pc; num_iterations::Integer=100, verbose::Bool=false, kwargs...)
    mf = normalize_mf_3d(mf)
    pc = normalize_mf_3d(pc)

    hyperparameters = model_hyperparameters(typeof(model))
    if length(hyperparameters) == 0
        return hyperparameters
    end

    keys_vec = collect(keys(hyperparameters))
    K = length(keys_vec)

    # ------------------------------------------------------------
    # FREEZE declared search bounds (never mutate these)
    # ------------------------------------------------------------
    declared = Dict{eltype(keys_vec), Tuple{Float64,Float64}}()
    for k in keys_vec
        hp = hyperparameters[k]
        declared[k] = (Float64(hp.low), Float64(hp.high))
    end

    # θ layout: [low1, width1, low2, width2, ...]
    lower = Vector{Float64}(undef, 2K)
    upper = Vector{Float64}(undef, 2K)
    θ_0   = Vector{Float64}(undef, 2K)

    for (i, k) in enumerate(keys_vec)
        (decl_lo, decl_hi) = declared[k]

        lower[2i] = lower[2i-1] = 0.0
        upper[2i] = upper[2i-1] = 1.0

        # Start at a random point in the range
        # 0 to 1
        θ_0[2i-1] = rand()
        θ_0[2i]   = rand() 
    end

    function mse(θ::AbstractVector)
        @assert length(θ) == 2K

        # ------------------------------------------------------------
        # Map (low, width) → (low, high) using frozen declared bounds
        # ------------------------------------------------------------
        for (i, k) in enumerate(keys_vec)
            u1 = θ[2i-1]
            u2 = θ[2i]
            L, H = declared[k]

            # Let u1 and u2 independently pick points in [L, H]
            # Then assign min to low and max to high.
            val1 = L + u1 * (H - L)
            val2 = L + u2 * (H - L)
            
            lo = min(val1, val2)
            hi = max(val1, val2)
            
            # Ensure a tiny epsilon width to prevent singular ranges
            if hi - lo < 1e-6
                hi = lo + 1e-6
            end

            hyperparameters[k].low  = lo
            hyperparameters[k].high = hi
        end

        # Ranges changed → rebuild model instance
        # Set our hyperparameters and rebuild the model
        error = 0
        num_replicates = 5
        for sample = 1:num_replicates
            init(model, mf, hyperparameters, rng=Random.MersenneTwister(sample))
            
            local y_hat
            try
                _, y_hat = fit_purkinje_cell_responses(model, mf, pc, max_iter=100)
            catch e
                if isa(e, LinearAlgebra.SingularException)
                    return Inf
                else
                    rethrow(e)
                end
            end
            error += mean((pc .- y_hat).^ 2) / num_replicates
        end
        verbose && println(θ, ": ", error)
        return error
    end

    # method = Optim.ParticleSwarm(; lower=lower, upper=upper, n_particles=num_particles)
    # options = Optim.Options(show_trace=false, show_every=10, iterations=num_iterations)
    # result = Optim.optimize(mse, θ_0, method, options)
    # verbose && println(result)
    # θ_final = result.minimizer
    search_range = [(lower[i], upper[i]) for i in 1:length(lower)]
    result = BlackBoxOptim.bboptimize(mse, θ_0; NumDimensions=length(lower), SearchRange=search_range, 
        MaxSteps=num_iterations, TraceInterval=(verbose ? 1.0 : Inf), TraceMode=(verbose ? :compact : :silent), 
        Method=:adaptive_de_rand_1_bin_radiuslimited, kwargs...); # separable_nes adaptive_de_rand_1_bin_radiuslimited
    verbose && println(result)
    θ_final = GrCModels.BlackBoxOptim.best_candidate(result)

    # ------------------------------------------------------------
    # Write back final solution using frozen bounds
    # ------------------------------------------------------------
    for (i, k) in enumerate(keys_vec)
        u1 = θ_final[2i-1]  # in [0,1]
        u2 = θ_final[2i]    # in [0,1]
        L, H = declared[k]
        val1 = L + u1 * (H - L)
        val2 = L + u2 * (H - L)

        hyperparameters[k].low  = min(val1, val2)
        hyperparameters[k].high = max(val1, val2)
    end

    return hyperparameters
end


"""
    test_generalization(model, mf, pc_responses; num_replicates=1)

Evaluate cross-condition generalization performance of a model by training on
individual conditions and testing on all conditions.

This function quantifies how well a model trained on one condition predicts
responses in other conditions. It constructs a condition-by-condition matrix
of coefficient of determination (R2) values, where each entry `(i, j)` reflects
performance when trained on condition `i` and evaluated on condition `j`.

# Overview
- Input tensors `mf` (inputs) and `pc_responses` (targets) are assumed to be
  3D arrays with dimensions corresponding to neurons × time × conditions.
- The model first computes a shared feature representation (`basis`) across
  all conditions.
- Target responses are mean-centered using a baseline window (first 100 timepoints).
- For each condition `i`:
  - Fit readout weights using only data from condition `i`.
  - Apply the learned weights to predict responses for all conditions `j`.
  - Compute R² between predicted and actual responses for each `(i, j)` pair.

# Arguments
- `model`: Model instance providing feature transformation (`basis`) and fitting routine.
- `mf::AbstractArray{<:Real,3}`: Input data (e.g., mossy fiber activity),
  shaped as `(neurons, time, conditions)`.
- `pc_responses::AbstractArray{<:Real,3}`: Target outputs (e.g., Purkinje cell responses),
  shaped as `(neurons, time, conditions)`.

# Returns
- `r2_matrix::Matrix{Float64}`: A square matrix of size `(num_conditions, num_conditions)`
  where entry `(i, j)` is the R2 value when training on condition `i` and testing on `j`.

"""
function test_generalization(model, mf::AbstractArray{<:Real, 3}, pc_responses::AbstractArray{<:Real, 3})
    @assert(size(mf, 2) == size(pc_responses, 2))
    @assert(size(mf, 3) == size(pc_responses, 3))
    
    # Create a training/test matrix, training on 1 condition, testing on the other conditions
    r2_matrix = zeros(size(mf, 3), size(mf, 3))
    granule_cell_responses = basis(model, mf)

    # Remove mean from the output
    pc = pc_responses .- mean(pc_responses[:, 1:100, :], dims=(2, 3))

    for i = 1:size(mf, 3)
        #W, Y_hat = nn_lsq(granule_cell_responses[:, :, i], pc[:, :, i])
        W, _ = fit_purkinje_cell_responses(model, mf[:, :, i:i], pc[:, :, i:i])

        for j = 1:size(mf, 3)
            Y_hat = W * granule_cell_responses[:, :, j];
            r2_matrix[i, j] = cor(dropdims(mean(Y_hat, dims=1), dims=1), dropdims(mean(pc[:, :, j], dims=1), dims=1)).^2
        end
    end
    return r2_matrix
end

"""
    Also called cross-validated R2, this computes the normalized
    mean squared error as 
    NMSE = sum(Y(t) - Y_hat(t)).^2 / sum(Y(t) - Y_bar(t)).^2
    Then our prediction R^2 = 1 - NMSE

This is also called "goodness of fit"
"""
function pooled_R2(Y_hat::AbstractMatrix, Y::AbstractMatrix)
    @assert size(Y_hat) == size(Y)
    y  = vec(Y)
    yh = vec(Y_hat)
    sse = sum((y .- yh).^2)
    sst = sum((y .- mean(y)).^2)   # baseline: global mean of the TEST data
    return 1 - sse / sst
end

function plot_generalization_matrix(r2_values::Matrix{<:Real}; speeds::AbstractVector=[10, 20, 30])
    fig = figure()
    pcolormesh(r2_values')
    xlabel("Training condition")
    ylabel("Test condition")
    colorbar(label="Shared variance r^2")
    for i = 1:size(r2_values, 1)
        for j = 1:size(r2_values, 2)
            text(i - 0.5, j - 0.5, rpad(string(round(r2_values[i, j], digits=2)), 2), horizontalalignment="center")
        end
    end

    xticks([1, 2, 3] .- 0.5, speeds)
    yticks([1, 2, 3] .- 0.5, speeds)
    clim(0, 1)
    return fig
end
function plot_generalization_matrix(r2_values::AbstractArray{<:Real, 3}; speeds::AbstractVector=[10, 20, 30])
    fig = plot_generalization_matrix(dropdims(mean(r2_values, dims=3), dims=3); speeds=speeds)
    for i = 1:size(r2_values, 1)
        for j = 1:size(r2_values, 2)
            quantiles = quantile(r2_values[i, j, :], [0.05, 0.95])
            s = string("[", rpad(string(round(quantiles[1], digits=2)), 2), ", ", rpad(string(round(quantiles[2], digits=2)), 2), "]")
            text(i - 0.5, j - 0.75, s, horizontalalignment="center")
        end
    end
    return fig
end

cosine_similarity(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}) = 
    LinearAlgebra.dot(x, y) / (LinearAlgebra.norm(x) * LinearAlgebra.norm(y))

"""
    cosine_similarity_generalization(model, mf, pc, train_index, test_index)

Return a list of the cosine similarities between weight matrices across two
conditions (one train, one test). We return a vector of the same length as
Purkinje cells - whose contents are the cosine similarity between the granule-to-
purkinje cell weight vector for that neuron across conditions.

Notes:
 - When `active_set` is true, we only use weights whose magnitude exceeds epsilon. This
   allows for a more robust
"""
function cosine_similarity_generalization(model::AbstractGRCBasisModel, mf::AbstractArray{<:Real, 3}, pc_responses::AbstractArray{<:Real, 3}; 
        train_index::Integer, test_index::Integer, active_set::Bool=false, ϵ::Real=1e-6)
    W_train, _ = fit_purkinje_cell_responses(model, mf[:, :, train_index], pc_responses[:, :, test_index])
    W_test, _ = fit_purkinje_cell_responses(model, mf[:, :, test_index], pc_responses[:, :, test_index])
    num_pcs = size(pc_responses, 1)
    cs = zeros(num_pcs)

    for i = 1:num_pcs
        if active_set
            select = abs.(W_train[i, :]) .> ϵ .|| abs.(W_test[i, :]) .> ϵ
        else
            select = Colon()
        end

        cs[i] = cosine_similarity(W_train[i, select], W_test[i, select])
    end
    return cs
end

function jaccard_similarity(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; ϵ::Real=1e-6)
    I_x = findall(abs.(x) .> ϵ)
    I_y = findall(abs.(y) .> ϵ)
    return length(intersect(I_x, I_y)) / length(union(I_x, I_y))
end


"""
    jaccard_similarity_generalization(model, mf, pc, train_index, test_index)

Return a list of the jaccard similarities between weight matrices across two
conditions (one train, one test). We return a vector of the same length as
Purkinje cells - whose contents are the cosine similarity between the granule-to-
purkinje cell weight vector for that neuron across conditions.
"""
function jaccard_similarity_generalization(model::AbstractGRCBasisModel, mf::AbstractArray{<:Real, 3}, pc_responses::AbstractArray{<:Real, 3}; 
        train_index::Integer, test_index::Integer, kwargs...)
    
    W_train, _ = fit_purkinje_cell_responses(model, mf[:, :, train_index], pc_responses[:, :, test_index])
    W_test, _ = fit_purkinje_cell_responses(model, mf[:, :, test_index], pc_responses[:, :, test_index])

    num_pcs = size(pc_responses, 1)
    ji = zeros(num_pcs)

    for i = 1:num_pcs
        ji[i] = jaccard_similarity(W_train[i, :], W_test[i, :])
    end
    return ji
end
