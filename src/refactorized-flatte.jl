# refactorized-flatte.jl
#
# Four-way refactor:
# (1) Full vs simpler model
# (2) Finite-width shift on/off (per channel)
# (3) Riemann sheet choice (± for each channel sqrt)
# (4) Style-B reparametrization Ef_corr -> Ef via shift_Ef / compute_corrected_Ef
#
# Assumptions:
# - Energies E are in MeV
# - Reduced masses μ, μ⁺ are in GeV
# - Threshold offset δ⁺ is in GeV (so δ⁺*1e3 is MeV)
# - D* widths ΓDˣ⁰, ΓDˣ⁺ are in GeV (so *1e3 is MeV)
#
# You likely have these in masseswidths.jl:
#   μ, μ⁺, δ⁺, ΓDˣ⁰, ΓDˣ⁺

# using Optim
# using NLSolve

# ----------------------------
# Types for the refactor knobs
# ----------------------------

const Sheet2 = NTuple{2,Int}          # (s1, s2), each ±1
const WidthFlags2 = NTuple{2,Bool}    # (w1, w2), each Bool

const PHYS_SHEET::Sheet2 = (+1, +1)
const NO_WIDTH::WidthFlags2 = (false, false)

# ----------------------------
# Model definitions
# ----------------------------

Base.@kwdef struct FlatteModel
    Ef_MeV::Float64
    g::Float64
    Γ₀_MeV::Float64
    fρ::Float64 = 0.0
    fω::Float64 = 0.0
end

Base.@kwdef struct FlatteModelSimpler
    Ef_MeV::Float64
    g::Float64
    Γ₀_MeV::Float64
end

"""
Style-B wrapper: fit Ef_corr_MeV, convert to Ef_MeV at evaluation time.

`base` must be a FlatteModel or FlatteModelSimpler (its Ef_MeV field is ignored).
"""
struct FlatteCorr{T}
    base::T
    Ef_corr_MeV::Float64
end

function ReparametrizeFlatteSimpler(nt)
    return FlatteCorr(
        FlatteModelSimpler(Ef_MeV = 0.0, g = nt.g, Γ₀_MeV = nt.Γ₀_MeV),
        nt.Ef_corr
    )
end

# ----------------------------
# Helpers for inelastic channels
# ----------------------------

inelastic_term(::FlatteModelSimpler, _E_MeV) = 0.0 + 0.0im

function inelastic_term(m::FlatteModel, E_MeV)
    # If you use ρ/ω terms, you must define BXρ and BXω somewhere.
    if (m.fρ != 0.0 || m.fω != 0.0)
        if !isdefined(@__MODULE__, :BXρ) || !isdefined(@__MODULE__, :BXω)
            error("BXρ/BXω not defined, but fρ/fω are nonzero. Define BXρ(E), BXω(E) or set fρ=fω=0.")
        end
    end
    term = 0.0 + 0.0im
    term += m.fρ * BXρ(E_MeV)
    term += m.fω * BXω(E_MeV)
    return term
end

# ----------------------------
# Unified momentum k(E) engine
# ----------------------------

_complex(x::Real) = complex(x, 0.0)
_complex(x::Complex) = x

"""
Nonrelativistic breakup momentum with a unified interface.

Arguments/units:
- E_MeV: energy relative to neutral threshold, in MeV (can be Real or Complex)
- μ_GeV: reduced mass, in GeV
Keyword args:
- δ_MeV: channel threshold offset, in MeV (0 for neutral, δ⁺*1e3 for charged)
- Γ_MeV: D* width used for complex-threshold shift, in MeV
- use_width: if true use E -> E + i Γ/2; else use +i0 regulator for real E
- s: sheet sign (+1 or -1), multiplies principal-branch expression
Convention matches your earlier code: k = i * sqrt(-2 μ * E[GeV]).
"""
function k_nr(E_MeV, μ_GeV; δ_MeV::Float64=0.0, Γ_MeV::Float64=0.0,
              use_width::Bool=false, s::Int=+1)
    Eeff = _complex(E_MeV - δ_MeV)
    if use_width
        Eeff += 0.5im * Γ_MeV
    else
        # Keep branch prescription for real energies (old +i0)
        if E_MeV isa Real
            Eeff += 1e-7im
        end
    end
    return s * (1im * sqrt(-2 * μ_GeV * (Eeff * 1e-3)))
end

# ----------------------------
# DD* loop/self-energy term
# ----------------------------

"""
DD* loop term Σ(E) used in the denominator:
  Σ(E) = 0.5 i g (k1 + k2)

Returns Complex in GeV units.
"""
function sigma_DDst(m, E_MeV;
                    sheet::Sheet2 = PHYS_SHEET,
                    widthflags::WidthFlags2 = NO_WIDTH,
                    Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    s1, s2 = sheet
    w1, w2 = widthflags
    Γ1_MeV, Γ2_MeV = Γs_MeV

    k1 = k_nr(E_MeV, μ;  δ_MeV=0.0,        Γ_MeV=Γ1_MeV, use_width=w1, s=s1)
    k2 = k_nr(E_MeV, μ⁺; δ_MeV=δ⁺ * 1e3,   Γ_MeV=Γ2_MeV, use_width=w2, s=s2)

    return 0.5im * m.g * (k1 + k2)
end

# ----------------------------
# Standard denominator (Ef is a physical parameter)
# ----------------------------

"""
Standard Flatté-like denominator for either model, evaluated with physical Ef_MeV.

Returns Complex in GeV units.

Structure:
D(E) = (E - Ef)*1e-3 + Σ_DDst(E) + 0.5 i Γ0 + 0.5 i * (fρ BXρ + fω BXω)   [full model only]
"""
function denominator_std(m::Union{FlatteModel,FlatteModelSimpler}, E_MeV;
                         sheet::Sheet2 = PHYS_SHEET,
                         widthflags::WidthFlags2 = NO_WIDTH,
                         Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    Σ = sigma_DDst(m, E_MeV; sheet=sheet, widthflags=widthflags, Γs_MeV=Γs_MeV)
    extra_inel = inelastic_term(m, E_MeV)  # 0 for simpler
    return (E_MeV - m.Ef_MeV) * 1e-3 +
           Σ +
           0.5im * (m.Γ₀_MeV * 1e-3) +
           0.5im * extra_inel
end

# Convenience amplitude
AJψππ(m::Union{FlatteModel,FlatteModelSimpler}, E_MeV; kwargs...) =
    1 / denominator_std(m, E_MeV; kwargs...)

# ----------------------------
# Style-B mapping: Ef_corr <-> Ef
# ----------------------------

"""
shift_Ef(g, Ef_corr) -> Ef (MeV)

Implements your Style-B reparametrization map, using a *calibration model*:
Ef=0, Γ0=0, fρ=fω=0 (DD* only), evaluated at E=Ef_corr.

By default uses physical sheet and no finite-width shift, matching the usual "convention"
for defining Ef_corr. You can override via kwargs.
"""
function shift_Ef(g::Float64, Ef_corr_MeV::Float64;
                  map_sheet::Sheet2 = PHYS_SHEET,
                  map_widthflags::WidthFlags2 = NO_WIDTH,
                  Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    calib = FlatteModelSimpler(Ef_MeV=0.0, g=g, Γ₀_MeV=0.0) # , fρ=0.0, fω=0.0
    D = denominator_std(calib, Ef_corr_MeV; sheet=map_sheet, widthflags=map_widthflags, Γs_MeV=Γs_MeV)
    return 1e3 * real(D)  # MeV
end

"""
compute_corrected_Ef(Ef, g) -> (; sol, Ef_corr_MeV)

Numerically inverts shift_Ef to find Ef_corr that corresponds to Ef.
"""
function compute_corrected_Ef(Ef_MeV::Float64, g::Float64; Ef_corr_guess=-0.04,
                              map_sheet::Sheet2 = PHYS_SHEET,
                              map_widthflags::WidthFlags2 = NO_WIDTH,
                              Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    sol = nlsolve(x -> (shift_Ef(g, x[1];
                                map_sheet=map_sheet,
                                map_widthflags=map_widthflags,
                                Γs_MeV=Γs_MeV) - Ef_MeV),
                  [Ef_corr_guess])
    return (; sol, Ef_corr_MeV=sol.zero[1])
end

# ----------------------------
# Convert Style-B wrapper to standard model, then evaluate
# ----------------------------

function to_standard(m::FlatteCorr{FlatteModel};
                     map_sheet::Sheet2 = PHYS_SHEET,
                     map_widthflags::WidthFlags2 = NO_WIDTH,
                     Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    Ef = shift_Ef(m.base.g, m.Ef_corr_MeV; map_sheet=map_sheet, map_widthflags=map_widthflags, Γs_MeV=Γs_MeV)
    return FlatteModel(Ef_MeV=Ef, g=m.base.g, Γ₀_MeV=m.base.Γ₀_MeV, fρ=m.base.fρ, fω=m.base.fω)
end

function to_standard(m::FlatteCorr{FlatteModelSimpler};
                     map_sheet::Sheet2 = PHYS_SHEET,
                     map_widthflags::WidthFlags2 = NO_WIDTH,
                     Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    Ef = shift_Ef(m.base.g, m.Ef_corr_MeV; map_sheet=map_sheet, map_widthflags=map_widthflags, Γs_MeV=Γs_MeV)
    return FlatteModelSimpler(Ef_MeV=Ef, g=m.base.g, Γ₀_MeV=m.base.Γ₀_MeV)
end

"""
Denominator for Style-B wrapper:
- First map Ef_corr -> Ef using map_* knobs (usually fixed convention)
- Then evaluate denominator_std at (sheet, widthflags) knobs
"""
function denominator(m::FlatteCorr{<:Union{FlatteModel,FlatteModelSimpler}}, E_MeV;
                     # mapping knobs (should usually be kept fixed)
                     map_sheet::Sheet2 = PHYS_SHEET,
                     map_widthflags::WidthFlags2 = NO_WIDTH,
                     # evaluation knobs
                     sheet::Sheet2 = PHYS_SHEET,
                     widthflags::WidthFlags2 = NO_WIDTH,
                     Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰ * 1e3, ΓDˣ⁺ * 1e3))
    mstd = to_standard(m; map_sheet=map_sheet, map_widthflags=map_widthflags, Γs_MeV=Γs_MeV)
    return denominator_std(mstd, E_MeV; sheet=sheet, widthflags=widthflags, Γs_MeV=Γs_MeV)
end

AJψππ(m::FlatteCorr{<:Union{FlatteModel,FlatteModelSimpler}}, E_MeV; kwargs...) = 1 / denominator(m, E_MeV; kwargs...)

# ----------------------------
# Pole search utility
# ----------------------------

"""
Find a pole by minimizing |D(E)|^2 in the complex plane.

- init: Complex initial guess in MeV
- kwargs forwarded to denominator/denominator_std depending on model type
"""
function pole_position(m, init::Complex = -0.1im * getfield(m isa FlatteCorr ? m.base : m, :Γ₀_MeV);
                       kwargs...)
    f(x) = begin
        E = x[1] + 1im*x[2]
        D = m isa FlatteCorr ? denominator(m, E; kwargs...) :
                               denominator_std(m, E; kwargs...)
        abs2(D)
    end
    fr = optimize(f, collect(reim(init)), BFGS())
    fr.minimum < 1e-8 || error("Pole not found: minimum = $(fr.minimum)")
    return complex(fr.minimizer...)  # MeV
end

#=# --- constructor typing ---
FlatteCorr(base::FlatteModel, Ef_corr) =
    FlatteCorr{FlatteModel}(base, Ef_corr)

FlatteCorr(base::FlatteModelSimpler, Ef_corr) =
    FlatteCorr{FlatteModelSimpler}(base, Ef_corr)

# --- standard model method ---
scattering_parameters(model::Union{FlatteModel,FlatteModelSimpler}) =
    scattering_parameters(typeof(model), model.Ef_MeV, model.g)=#

# --- style B method ---
function scattering_parameters(model::FlatteCorr{<:Union{FlatteModel,FlatteModelSimpler}};
                               map_sheet::Sheet2 = PHYS_SHEET,
                               map_widthflags::WidthFlags2 = NO_WIDTH,
                               Γs_MeV::Tuple{Float64,Float64} = (ΓDˣ⁰*1e3, ΓDˣ⁺*1e3))

    mstd = to_standard(model;
                       map_sheet = map_sheet,
                       map_widthflags = map_widthflags,
                       Γs_MeV = Γs_MeV)

    return scattering_parameters(mstd)
end