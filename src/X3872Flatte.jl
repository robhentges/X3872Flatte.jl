module X3872Flatte

import Base: denominator
using Parameters
using NLsolve
using QuadGK
using Optim

include("masseswidths.jl")

export FlatteModel, FlatteModelSimpler
export shift_Ef, shift_Ef_simpler
export ReparametrizeFlatte, ReparametrizeFlatteSimpler
export compute_corrected_Ef, compute_corrected_Ef_simpler
export AJψππ, denominator, denominator_cont
export scattering_parameters
export pole_position, pole_position_cont
include("flatte.jl")

include("branchings.jl")

include("plausible_parameters.jl")

end # module
