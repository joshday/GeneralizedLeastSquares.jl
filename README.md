# GeneralizedLeastSquares

This package provides highly optimized solvers for the [generalized least squares](https://en.wikipedia.org/wiki/Generalized_least_squares) problem:

For matrices `X` and `W`, vector `y` of appropriate dimensions:

- Objective Function: `argmin(β) (y - Xβ)'W(y - Xβ)`
- [Normal Equation](https://mathworld.wolfram.com/NormalEquation.html): `X'WXβ = X'Wy`

## Use Cases (Pardon the ambiguity between pseudocode and math)

- Ordinary Least Squares (OLS): `W = LinearAlgebra.I`
- Weighted Least Squares (WLS): `W = inv(Diagonal(weights))`
- Aitken Model: `y = Xβ + ϵ` where `Var(ϵ) = σ²V` for some `V`.  In this case, set `W = inv(V)`.
    - Heteroskedasticity (see WLS): `V = Diagonal(σ²₁, ..., σ²ₙ)`
    - [Autoregressive model - AR(1)](https://en.wikipedia.org/wiki/Autoregressive_model):
        - Model: `yₜ = C + ρ yₜ₋₁ + ϵₜ`
        - `Var(yₜ) = V = Var(ϵₜ) / (1 - ρ^2) .* [ρ ^ abs(i-j) for i in 1:n, j in 1:n]`
    - Equicorrelation Model: `Var(ϵ) = σ²I .+ τ`


## Solvers

Firstly, why even provide multiple solvers?

**There is no universally superior method**.  Your task may prioritize stability over speed (or vice versa).

### `SweepGLS(x, y, w)`

- GLS via the [sweep operator](https://github.com/joshday/SweepOperator.jl)
- Less stable, also gets you standard errors for free.

### `CholeskyGLS(x, y, w)`

- GLS via the cholesky decomposition.
- Less stable, fastest.

### `QR_GLS(x, y, sqrt_w)`

- GLS via the QR decomposition (most stable, but much slower for n >> p).
- Most stable.
- Note that in order to be more efficient, this algorithm uses the matrix square root as the third argument.

## Benchmarks (see `test/simulations.jl`)

Calculated on:
```
julia> versioninfo()
Julia Version 1.7.0
Commit 3bf9d17731 (2021-11-30 12:12 UTC)
Platform Info:
  OS: macOS (arm64-apple-darwin21.1.0)
  CPU: Apple M1 Pro
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, cyclone)
```

Data generation (for given `n::Int`, `p::Int`, v::Matrix):

```julia
x = randn(n, p)
errs = w isa UniformScaling ? randn(n) : cholesky(w) * randn(n)
y = x * (1:p) + errs
```

Included in benchmarks:
```
linreg(x, y) = Hermitian(x'x) \ x'y
linreg(x, y, ::UniformScaling) = linreg(x, y)
linreg(x, y, w) = (A = x'w; Hermitian(A*x) \ A*y)
```

```
Simulation Settings: (n=100000, p=100, w isa LinearAlgebra.UniformScaling{Bool})

 model        time        gctime    memory      allocs
───────────────────────────────────────────────────────
 SweepGLS     16.205 ms   0.000 ns  81.59 KiB   6
 CholeskyGLS  14.117 ms   0.000 ns  160.58 KiB  9
 QR_GLS       156.134 ms  2.347 ms  154.33 MiB  15
 linreg       13.761 ms   0.000 ns  209.09 KiB  9


--------------------------------------------------------------
Simulation Settings: (n=100000, p=100, w isa 100000×100000 LinearAlgebra.Diagonal{Float64, Vector{Float64}})

 model        time        gctime      memory      allocs
─────────────────────────────────────────────────────────
 SweepGLS     43.760 ms   687.542 μs  76.53 MiB   16
 CholeskyGLS  41.049 ms   704.250 μs  76.61 MiB   19
 QR_GLS       158.903 ms  3.463 ms    231.38 MiB  17
 linreg       64.699 ms   2.360 ms    152.74 MiB  12


--------------------------------------------------------------
Simulation Settings: (n=100000, p=1000, w isa LinearAlgebra.UniformScaling{Bool})

 model        time        gctime    memory     allocs
──────────────────────────────────────────────────────
 SweepGLS     442.315 ms  0.000 ns  7.66 MiB   6
 CholeskyGLS  351.730 ms  0.000 ns  15.30 MiB  9
 QR_GLS       2.595 s     7.009 ms  1.51 GiB   15
 linreg       356.247 ms  0.000 ns  15.77 MiB  9


--------------------------------------------------------------
Simulation Settings: (n=100000, p=1000, w isa 100000×100000 LinearAlgebra.Diagonal{Float64, Vector{Float64}})

 model        time        gctime     memory      allocs
────────────────────────────────────────────────────────
 SweepGLS     984.948 ms  9.941 ms   785.87 MiB  16
 CholeskyGLS  878.098 ms  10.097 ms  793.51 MiB  19
 QR_GLS       2.561 s     17.219 ms  2.25 GiB    17
 linreg       1.668 s     21.364 ms  1.51 GiB    12
```

## Resources

- [Hua Zhou](http://hua-zhou.github.io)'s Course Notes at NC State:
    - [ST 758 - Statistical Computing - Fall 2014 (PDF)](http://hua-zhou.github.io/teaching/st758-2014fall/ST758-2014-Fall-LecNotes.pdf)
    - [ST 552 - Linear Models - Fall 2013 (PDF)](http://hua-zhou.github.io/teaching/st552-2013fall/ST552-2013-Fall-LecNotes.pdf)
- Lange, K (2010). *Numerical analysis for statisticians*. New York: Springer.
