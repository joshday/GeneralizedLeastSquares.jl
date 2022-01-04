# GeneralizedLeastSquares

This package provides solvers for the [generalized least squares](https://en.wikipedia.org/wiki/Generalized_least_squares) problem:

For matrices `X` and `W`, vector `y` of appropriate dimensions:

- Objective Function:

```
argmin(β) (y - Xβ)'W(y - Xβ)
```

- Normal Equations:

```
X'WXβ = X'Wy
```

Note: Solvers have also been optimized for the case when `W = I`.

## Solvers

### `SweepGLS`

- GLS via the [sweep operator](https://github.com/joshday/SweepOperator.jl)

### `CholeskyGLS`

- GLS via the cholesky decomposition.

### `QR_GLS`

- GLS via the QR decomposition (most stable, but much slower for n >> p).

## Resources

- http://hua-zhou.github.io/teaching/st758-2014fall/ST758-2014-Fall-LecNotes.pdf
