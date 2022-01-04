module GLSBenchmarks

using GeneralizedLeastSquares: coef, SweepGLS, CholeskyGLS, QR_GLS
using LinearAlgebra
using BenchmarkTools
using Statistics
using DataFrames

linreg(x, y) = Hermitian(x'x) \ x'y
linreg(x, y, ::UniformScaling) = linreg(x, y)
linreg(x, y, w) = (A = x'w; Hermitian(A*x) \ A*y)




function sim(n, p, w=I)
    x = randn(n, p)
    errs = w isa UniformScaling ? randn(n) : Matrix(cholesky(w).U) * randn(n)
    y = x * (1:p) + errs
    println("--------------------------------------------------------------")
    println("Simulation Settings: (n=$n, p=$p, w isa $(summary(w)))")
    println()

    results = DataFrame(model=[], time=[], gctime=[], memory=[], allocs=[])

    for model in [SweepGLS, CholeskyGLS, QR_GLS, linreg]
        @info "Running $model..."
        b = median(@benchmark $model($x, $y, $w))
        push!(results, (
            model=replace(string(model), "GeneralizedLeastSquares." => ""),
            time=BenchmarkTools.prettytime(b.time),
            gctime=BenchmarkTools.prettytime(b.gctime),
            memory=BenchmarkTools.prettymemory(b.memory),
            allocs=b.allocs
        ))
    end

    show(results, show_row_number=false, summary=false, eltypes=false)
    println()
    println()
end

for i in 5, j in 2:3
    sim(10^i, 10^j, I)
    sim(10^i, 10^j, inv(Diagonal(rand(10^i))))
end


end
