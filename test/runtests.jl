using GeneralizedLeastSquares: coef, LinRegQR, LinRegCholesky, LinRegBunchKaufman, LinRegSweep
using LinearAlgebra
using Test

n, p = 10^6, 50

x = randn(n, p)

y = x * (1:50) + randn(n)

w = Diagonal(rand(n))

β̂ = x \ y

β̂2 = (x' * w * x) \ (x' * w * y)

@testset "OLS" begin
    @testset "LinRegQR" begin
        model = LinRegQR(x, y)
        @test coef(model) ≈ β̂
    end
    @testset "LinRegCholesky" begin
        model = LinRegCholesky(x, y)
        @test coef(model) ≈ β̂
    end
    @testset "LinRegBunchKaufman" begin
        model = LinRegBunchKaufman(x, y)
        @test coef(model) ≈ β̂
    end
    @testset "LinRegSweep" begin
        model = LinRegSweep(x, y)
        @test coef(model) ≈ β̂
    end
end
@testset "GLS" begin
    @testset "LinRegQR" begin
        wsqrt = sqrt(w)
        x2 = wsqrt * x
        y2 = wsqrt * y
        model = LinRegQR(x2, y2)
        @test coef(model) ≈ β̂2
    end
    @testset "LinRegCholesky" begin
        model = LinRegCholesky(x, y, w)
        @test coef(model) ≈ β̂2
    end
    @testset "LinRegBunchKaufman" begin
        model = LinRegBunchKaufman(x, y, w)
        @test coef(model) ≈ β̂2
    end
    @testset "LinRegSweep" begin
        model = LinRegSweep(x, y, w)
        @test coef(model) ≈ β̂2
    end
end
