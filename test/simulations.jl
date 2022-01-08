module GLSBenchmarks

using GeneralizedLeastSquares: coef, LinRegQR, LinRegCholesky, LinRegBunchKaufman, LinRegSweep
using LinearAlgebra
using BenchmarkTools
using Statistics
using UnicodePlots
using OrderedCollections: OrderedDict



BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

function sim(n, p, w=nothing)
    title="n=$n, p=$p, w=$(typeof(w))"
    x = randn(n, p)
    errs = isnothing(w) ? randn(n) : cholesky(w).U * randn(n)
    y = x * (1:p) + errs

    args = isnothing(w) ? (x, y) : (x, y, w)

    out = OrderedDict{String, BenchmarkTools.Trial}()

    out["LinRegSweep"] =        @benchmark LinRegSweep($args...)
    out["LinRegQR"] =           @benchmark LinRegQR($args...)
    out["LinRegCholesky"] =     @benchmark LinRegCholesky($args...)
    out["LinRegBunchKaufman"] = @benchmark LinRegBunchKaufman($args...)


    data = round.(map(x -> median(x).time, values(out)) ./ 10^9, digits=4)
    b = barplot(collect(keys(out)), data; width=70, title)
    show(b)
    println()
end

sim(10^6, 50)
sim(10^6, 50, Diagonal(rand(10^6)))

sim(1000, 500)
sim(1000, 500, Diagonal(rand(1000)))


end
