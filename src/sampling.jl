abstract type AbstractRandomRange end

import Random.AbstractRNG
"""
    UniformSpec(low, high): sample uniformly on [low, high].
"""
mutable struct Uniform <: AbstractRandomRange
    low::Float64
    high::Float64
    function Uniform(low::Real, high::Real)
        lowf = Float64(low); highf = Float64(high)
        new(lowf, highf)
    end
end

function sample(rng::AbstractRNG, s::Uniform)
    lo = min(s.low, s.high)
    hi = max(s.low, s.high)
    rand(rng) * (hi - lo) + lo
end

function Base.show(io::IO, s::Uniform)
    print(io, "$(typeof(s)) from [$(s.low), $(s.high)]")
end


"""
    LogUniformSpec(low, high): sample log-uniform on [low, high]. Requires low, high > 0.
"""
mutable struct LogUniform <: AbstractRandomRange
    low::Float64
    high::Float64
    function LogUniform(low::Real, high::Real)
        lowf = Float64(low); highf = Float64(high)
        new(lowf, highf)
    end
end

function Base.show(io::IO, s::LogUniform)
    print(io, "$(typeof(s)) from [$(s.low), $(s.high)]")
end

function sample(rng::AbstractRNG, s::LogUniform)
    lo = min(s.low, s.high)
    hi = max(s.low, s.high)
    lo = max(1e-6, lo)
    hi = max(1e-6, hi)

    lo = log(lo)
    hi = log(hi)
    return exp(rand(rng) * (hi - lo) + lo)
end