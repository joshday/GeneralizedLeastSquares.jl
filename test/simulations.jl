module GLSBenchmarks

using GeneralizedLeastSquares: SweepGLS, CholeskyGLS, QR_GLS
using LinearAlgebra
using BenchmarkTools
using Statistics
using DataFrames
using UnicodePlots
using OrderedCollections: OrderedDict



BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5


linreg(x, y) = Hermitian(x'x) \ x'y
linreg(x, y, ::UniformScaling) = linreg(x, y)
linreg(x, y, w) = (A = x'w; Hermitian(A*x) \ A*y)


function sim(n, p, w=I)
    x = randn(n, p)
    errs = w isa UniformScaling ? randn(n) : Matrix(cholesky(w).U) * randn(n)
    y = x * (1:p) + errs

    out = OrderedDict{String, BenchmarkTools.Trial}()

    out["SweepGLS"] =       @benchmark SweepGLS($x, $y, $w)
    out["CholeskyGLS"] =    @benchmark CholeskyGLS($x, $y, $w)
    out["QR_GLS"] =         @benchmark QR_GLS($x, $y, $w)
    out["linreg"] =         @benchmark linreg($x, $y, $w)

    df = DataFrame(model="SweepGLS", times=out["SweepGLS"])
    append!(df, DataFrame(model="CholeskyGLS", times=out["CholeskyGLS"]))
    append!(df, DataFrame(model="QR_GLS", times=out["QR_GLS"]))
    append!(df, DataFrame(model="linreg", times=out["linreg"]))


    wname = w isa UniformScaling ? "I" : typeof(w)

    data = round.(map(x -> median(x).time, values(out)) ./ 10^9, digits=4)
    b = barplot(collect(keys(out)), data, width=80, title="n=$n, p=$p, w=$wname")
    show(b)
    println()
end

for i in 4:5, j in 2:3
    sim(10^i, 10^j, I)
    sim(10^i, 10^j, inv(Diagonal(rand(10^i))))
end


end
