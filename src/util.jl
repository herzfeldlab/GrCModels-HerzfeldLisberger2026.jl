"""
    normalize_mf_3d(mf) -> X::Array{Float64,3}

Acceptable inputs:
- mf::AbstractMatrix               (MF × Time) -> MF × Time × 1
- mf::AbstractArray{<:Real,3}      (MF × Time × Condition) -> Float64 copy
- mf::AbstractVector{<:AbstractMatrix}  (vector of MF × Time) -> MF × Time × Condition
    - requires all elements have identical size (MF, Time)

Returns:
- X::Array{Float64,3} with shape (MF, Time, Condition)
"""
function normalize_mf_3d(mf)::Array{Float64,3}
    if mf isa AbstractMatrix
        MF, T = size(mf)
        X = Array{Float64,3}(undef, MF, T, 1)
        X[:, :, 1] .= mf
        return X

    elseif mf isa AbstractArray{<:Real,3}
        # Enforce Float64 storage for downstream consistency/perf
        return Float64.(mf)

    elseif mf isa AbstractVector{<:AbstractMatrix}
        length(mf) > 0 || error("normalize_mf_3d: empty Vector input is not allowed.")

        MF = size(mf[1], 1)
        T  = size(mf[1], 2)
        C  = length(mf)

        # Require consistent dimensions so we can return a single 3D array
        for (i, m) in pairs(mf)
            size(m, 1) == MF || error("normalize_mf_3d: element $i has MF=$(size(m,1)), expected $MF.")
            size(m, 2) == T  || error("normalize_mf_3d: element $i has Time=$(size(m,2)), expected $T.")
        end

        X = Array{Float64,3}(undef, MF, T, C)
        for c in 1:C
            X[:, :, c] .= mf[c]
        end
        return X

    else
        error("normalize_mf_3d: unsupported input type $(typeof(mf)).")
    end
end

"""
    apply_warmup(X; warmup=0, warmup_mode=:repeat_first) -> Xw

X must be MF × Time × Condition. Returns MF × (Time+warmup) × Condition.
"""
function apply_warmup(X::AbstractArray{<:Real,3}; warmup::Integer=0, warmup_mode=:repeat_first)
    warmup <= 0 && return Float64.(X)

    MF, T, C = size(X)
    Xw = Array{Float64,3}(undef, MF, T + warmup, C)

    if warmup_mode == :repeat_first
        Xw[:, 1:warmup, :] .= reshape(Float64.(X[:, 1, :]), MF, 1, C)
    elseif warmup_mode == :zeros
        Xw[:, 1:warmup, :] .= 0.0
    else
        error("apply_warmup: unknown warmup_mode=$warmup_mode (expected :repeat_first or :zeros).")
    end

    Xw[:, warmup+1:end, :] .= Float64.(X)
    return Xw
end