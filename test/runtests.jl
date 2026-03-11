using X3872Flatte
using Test

@testset "Testing amplitude" begin
    model = FlatteModel(
        Ef_MeV = -7.5, g = 0.1, Γ₀_MeV = 1.8,
        fρ = 0.0, fω = 0.0) # Ef is above the second threshold
    # 
    Ai = AJψππ(model, 1.1)
    @test Ai ≈ 154.17507563874798 - 179.56915419693846im
end

@testset "Testing simpler amplitude" begin
    model = FlatteModelSimpler(
        Ef_MeV = -7.5, g = 0.1, Γ₀_MeV = 1.8) # Ef is above the second threshold
    # 
    Ai = AJψππ(model, 1.1)
    @test Ai ≈ 154.17507563874798 - 179.56915419693846im
end

@testset "Pole position" begin
    model = FlatteModel(Ef_MeV = -8.8, g = 0.13, Γ₀_MeV = 1.8, fρ = 0.0, fω = 0.0)
    E_pole = pole_position(model)
    @test isapprox(E_pole, 0.0134358620 - 0.1152417753im, atol = 1e-8)
end

@testset "Pole position continued" begin
    model = FlatteModelSimpler(Ef_MeV = -8.8, g = 0.13, Γ₀_MeV = -1.8)
    E_pole = pole_position_cont(model)
    @test isapprox(E_pole, -4.798914249890528 + 1.200331014099233im, atol = 1e-8)
end


@testset "Reparametrize Flatte" begin
    g, Γ₀_MeV, Ef_MeV = 0.13, 1.8, -8.8
    _, Ef_corr = compute_corrected_Ef(Ef_MeV, g)
    model = ReparametrizeFlatte((; g, Γ₀_MeV, Ef_corr, fρ = 0, fω = 0))
    @test model.Ef_MeV ≈ Ef_MeV
end

@testset "Reparametrize Flatte Simpler" begin
    g, Γ₀_MeV, Ef_MeV = 0.13, 1.8, -8.8
    _, Ef_corr = compute_corrected_Ef_simpler(Ef_MeV, g)
    model = ReparametrizeFlatteSimpler((; g, Γ₀_MeV, Ef_corr))
    @test model.Ef_MeV ≈ Ef_MeV
end

