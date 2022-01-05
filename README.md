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

### Timings (in seconds)
```
                                               n=10000, p=100, w=I
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■ 0.0037
   CholeskyGLS ┤■■■■■■■ 0.0018
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0174
        linreg ┤■■■■■■■ 0.0016
               └                                                                                ┘
                              n=10000, p=100, w=Diagonal{Float64, Vector{Float64}}
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0073
   CholeskyGLS ┤■■■■■■■■■■■■■■■■■ 0.0045
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0194
        linreg ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0079
               └                                                                                ┘
                                              n=10000, p=1000, w=I
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.1682
   CholeskyGLS ┤■■■■■■■■■■■■■■ 0.0618
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.3258
        linreg ┤■■■■■■■■■■■■■■■■■■■ 0.0875
               └                                                                                ┘
                              n=10000, p=1000, w=Diagonal{Float64, Vector{Float64}}
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.3292
   CholeskyGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.1837
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.4683
        linreg ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.3575
               └                                                                                ┘
                                              n=100000, p=100, w=I
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■ 0.0255
   CholeskyGLS ┤■■■■■■■ 0.0217
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.2253
        linreg ┤■■■■■■ 0.0184
               └                                                                                ┘
                              n=100000, p=100, w=Diagonal{Float64, Vector{Float64}}
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■■ 0.0437
   CholeskyGLS ┤■■■■■■■■■■■■■■■■ 0.0441
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.1966
        linreg ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0877
               └                                                                                ┘
                                              n=100000, p=1000, w=I
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■ 0.7016
   CholeskyGLS ┤■■■■■■■■■■■■ 0.5973
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 3.5877
        linreg ┤■■■■■■■■■■■■■ 0.6376
               └                                                                                ┘
                             n=100000, p=1000, w=Diagonal{Float64, Vector{Float64}}
               ┌                                                                                ┐
      SweepGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■ 1.4195
   CholeskyGLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 1.6149
        QR_GLS ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 4.0976
        linreg ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 3.2898
               └                                                                                ┘
```

## Resources

- [Hua Zhou](http://hua-zhou.github.io)'s Course Notes at NC State:
    - [ST 758 - Statistical Computing - Fall 2014 (PDF)](http://hua-zhou.github.io/teaching/st758-2014fall/ST758-2014-Fall-LecNotes.pdf)
    - [ST 552 - Linear Models - Fall 2013 (PDF)](http://hua-zhou.github.io/teaching/st552-2013fall/ST552-2013-Fall-LecNotes.pdf)
- Lange, K (2010). *Numerical analysis for statisticians*. New York: Springer.
