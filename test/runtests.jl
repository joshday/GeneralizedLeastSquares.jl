using GeneralizedLeastSquares: SweepGLS, CholeskyGLS, QR_GLS, coef
using LinearAlgebra
using Test

@testset "GeneralizedLeastSquares.jl" begin
    n, p = 10^6, 50
    x = randn(n, p)
    y = x * (1:50) + randn(n)
    v = Diagonal(rand(n))
    β = x \ y
    β2 = (x' * v * x) \ (x' * v * y)


    @testset "SweepGLS" begin
        m = SweepGLS(x, y)
        @test coef(m) ≈ β
        m2 = SweepGLS(x, y, v)
        @test coef(m2) ≈ β2
    end
    @testset "CholeskyGLS" begin
        m = CholeskyGLS(x, y)
        @test coef(m) ≈ β
        m2 = CholeskyGLS(x, y, v)
        @test coef(m2) ≈ β2
    end
    @testset "QR_GLS" begin
        m = QR_GLS(x, y)
        @test coef(m) ≈ β
        m2 = QR_GLS(x, y, sqrt(v))
        @test coef(m2) ≈ β2
    end
end
