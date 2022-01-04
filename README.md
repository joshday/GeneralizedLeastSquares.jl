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

### `SweepGLS(x, y, w)`

- GLS via the [sweep operator](https://github.com/joshday/SweepOperator.jl)

### `CholeskyGLS(x, y, w)`

- GLS via the cholesky decomposition.

### `QR_GLS(x, y, sqrt_w)`

- GLS via the QR decomposition (most stable, but much slower for n >> p).
- Note that in order to be more efficient, this algorithm uses the matrix square root as the third argument.

## Resources

- [Hua Zhou](http://hua-zhou.github.io)'s Course Notes at NC State:
    - [ST 758 - Statistical Computing - Fall 2014 (PDF)](http://hua-zhou.github.io/teaching/st758-2014fall/ST758-2014-Fall-LecNotes.pdf)
    - [ST 552 - Linear Models - Fall 2013 (PDF)](http://hua-zhou.github.io/teaching/st552-2013fall/ST552-2013-Fall-LecNotes.pdf)
- Lange, K (2010). *Numerical analysis for statisticians*. New York: Springer.
