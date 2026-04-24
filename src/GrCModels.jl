module GrCModels

using Statistics
using StatsBase
using PythonPlot
using LinearAlgebra

include("sampling.jl")
include("interface.jl");
include("util.jl");
include("fitting.jl")
include("passthrough_model.jl");
include("nonlinear_passthrough_model.jl");
include("gilmer_model.jl")
include("random_projection_model.jl")
include("stp_model.jl")
include("explicit_basis.jl")

# Re-export
using .GrCModelsInterface: model_hyperparameters, basis
export model_hyperparameters, basis

end # module GrcModels
