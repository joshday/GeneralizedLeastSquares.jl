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

- `LinRegQR(x, y)`
   - Most stable.
   - Used by R/GLM.jl
   - Essentially: `qr(x) \ y `
- `LinRegSweep(x, y, w=I)`
   - Used by SAS
   - Does the heavy computation of standard errors (`inv(x' * w * x)`) for free.
   - Essentially: `SweepOperator.sweep([x y]' * w * [x y], 1:size(x, 2))`
- `LinRegCholesky(x, y, w=I)`
   - Essentially: `cholesky(Hermitian(x'*w*x)) \ x'*w*y`
- `LinRegBunchKaufman(x, y, w=I)`
   - Essentially: `bunchkaufman(Hermitian(x'*w*x)) \ x'*w*y`

## Benchmarks (see `test/simulations.jl`)

### Calculated on:
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

### Data generation (for given `n::Int`, `p::Int`):

```julia
x = randn(n, p)
errs = cholesky(w).U * randn(n)  # For ws: I(n) and Diagonal(rand(n))
y = x * (1:p) + errs
```

### OLS Timings

```
                                                  n=1000000, p=50, w=Nothing
                      ┌                                                                                ┐
          LinRegSweep ┤■■■■■■■■■■■■■■ 0.1423
             LinRegQR ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.7208
       LinRegCholesky ┤■■■■■■■■■■■■■■ 0.1421
   LinRegBunchKaufman ┤■■■■■■■■■■■■■■ 0.1414
                      └                                                                                ┘
                              n=1000000, p=50, w=LinearAlgebra.Diagonal{Float64, Vector{Float64}}
                      ┌                                                                                ┐
          LinRegSweep ┤■■■■■■■■■■■■■■■■ 0.1696
             LinRegQR ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.7722
       LinRegCholesky ┤■■■■■■■■■■■■■■■ 0.1659
   LinRegBunchKaufman ┤■■■■■■■■■■■■■■■ 0.164
                      └                                                                                ┘
                                                   n=1000, p=500, w=Nothing
                      ┌                                                                                ┐
          LinRegSweep ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.0295
             LinRegQR ┤■■■■■■■■■■■■■■■■■■■■■■ 0.0092
       LinRegCholesky ┤■■■■■■ 0.0024
   LinRegBunchKaufman ┤■■■■■■■■■■ 0.0043
                      └                                                                                ┘
                               n=1000, p=500, w=LinearAlgebra.Diagonal{Float64, Vector{Float64}}
                      ┌                                                                                ┐
          LinRegSweep ┤■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ 0.03
             LinRegQR ┤■■■■■■■■■■■■■■■■■■■■■■■■ 0.0097
       LinRegCholesky ┤■■■■■■■■■ 0.0038
   LinRegBunchKaufman ┤■■■■■■■■■■■■■■ 0.0057
```

## Resources

- [Hua Zhou](http://hua-zhou.github.io)'s Course Notes:
   - [UCLA - Biostat 257 - Statistical Computing - Spring 2021](https://ucla-biostat-257-2021spring.github.io/syllabus/syllabus.html)
   - [NC State - ST 758 - Statistical Computing - Fall 2014](http://hua-zhou.github.io/teaching/st758-2014fall/schedule.html)
   - [NC State - ST 552 - Linear Models - Fall 2013](http://hua-zhou.github.io/teaching/st552-2013fall/schedule.html)
- Lange, K (2010). *Numerical analysis for statisticians*. New York: Springer.
