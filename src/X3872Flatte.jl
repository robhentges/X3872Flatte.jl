module X3872Flatte

import Base: denominator
using Parameters
using NLsolve
using QuadGK
using Optim

include("masseswidths.jl")

export FlatteModel, FlatteModelSimpler, FlatteCorr
export inelastic_term, k_nr, sigma_DDst, denominator_std
export shift_Ef, compute_corrected_Ef, to_standard
export AJψππ, denominator, pole_position
export scattering_parameters
include("refactorized-flatte.jl")

include("branchings.jl")

include("plausible_parameters.jl")

end # module
