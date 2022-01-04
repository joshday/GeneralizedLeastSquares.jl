module GeneralizedLeastSquares

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasComplex
using SweepOperator: sweep!
using Statistics: mean

#-----------------------------------------------------------------------------# Algorithm
abstract type Algorithm end

coef(alg::Algorithm) = alg.β

function Base.show(io::IO, g::T) where {T<:Algorithm}
    nm = replace(string(T), "GeneralizedLeastSquares." => "")
    print(IOContext(io, :compact=>true), "$nm with coef: ", coef(g)')
end

docstring = """
solves the generalized least squares problem:

- Objective Function:

``math
argmin_β (y - Xβ)'W (y - Xβ)
``

- Normal equations:
``math
X'WXβ = X'Wy
``

### Numerical Stability

In order of stability (least-to-most stable):

- [`SweepGLS`](@ref)
- [`CholeskyGLS`](@ref)
- [`QR_GLS`](@ref)
"""

#-----------------------------------------------------------------------------# augmat
"""
    augmat!(a, x, y, W=I)

Overwrite `a` with the "augmented" block matrix (upper triangle only):

```
| X'WX   X'Wy |
| y'WX   y'Wy |  ./ n
```
"""
function augmat!(a, x::StridedMatrix{T}, y::StridedVector{T}, ::UniformScaling=I) where {T<:Union{BlasFloat, BlasComplex}}
    n, p = size(x)
    α = T(1 / n)
    @views @inbounds begin
        BLAS.syrk!('U', 'T', α, x, zero(T), a[1:p, 1:p]) # x'x
        BLAS.gemv!('T', α, x, y, zero(T), a[1:p, end])   # x'y
        a[end] = mean(abs2, y)                           # y'y
    end
    return a
end

function augmat!(a, x, y, v_inv)
    n, p = size(x)
    @views @inbounds begin
        xtv = x'v_inv
        a[1:p, 1:p] = xtv * x ./ n      # X'WX
        a[1:p, end] = xtv * y ./ n      # X'Wy
        a[end] = dot(y, v_inv, y) / n   # y'Wy
    end
    return a
end

"""
    augmat(x, y, W=I)

Create the "augmented" block matrix (upper triangle only):

```
| X'WX   X'Wy |
| y'WX   y'Wy |  ./ n
```

- See [augmat!](@ref) for the in-place version.
"""
function augmat(x, y, v_inv)
    p = size(x,1) + 1
    a = Matrix{promote_type(eltype(x), eltype(y), eltype(v_inv))}(undef, p, p)
    augmat!(a, x, y, v_inv)
end

#-----------------------------------------------------------------------------# SweepGLS
"""
    SweepGLS(x, y, v_inv)

Using the [sweep operator](https://github.com/joshday/SweepOperator.jl), `SweepGLS` $docstring

### Additional Notes

- The sweep operator is how SAS performs regression.
- The inverse of `x' * W * x ./ n` is provided (`model.matrix[1:end-1, 1:end-1]`).
    - Required for standard errors of β.
- The biased mean squared error `MSE = mean(abs2, y - x*β)` is provided (`model.matrix[end]`).
"""
struct SweepGLS{T} <: Algorithm
    matrix::Matrix{T}
    β::Vector{T}
end

function SweepGLS(x, y, v_inv=I)
    a = augmat(x, y, v_inv)
    sweep!(a, 1:size(x,2))
    SweepGLS(a, a[1:end-1, end])
end

#-----------------------------------------------------------------------------# CholeskyGLS
"""
    CholeskyGLS(x, y, W=I)

Using the cholesky decomposition, `CholeskyGLS` $docstring

### Additional Notes

- The biased mean squared error `mean(abs2, y - x*β)` is easily available:
    - `dot(model.decomp.U[1:end-1, end])`
- If standard errors are needed, `inv(x' * W * x ./ n)` is available via `inv(model.decomp)`.
"""
struct CholeskyGLS{T<:Cholesky, S} <: Algorithm
    decomp::T
    β::Vector{S}
end

function CholeskyGLS(x, y, v_inv=I)
    a = augmat(x, y, v_inv)
    decomp = cholesky(Hermitian(a, :U))
    β = @views decomp.U[1:end-1, 1:end-1] \ decomp.U[1:end-1, end]
    CholeskyGLS(decomp, β)
end

#-----------------------------------------------------------------------------# QR_GLS
"""
    QR_GLS(x, y, W=I)

Using the cholesky decomposition, `QR_GLS` $docstring

### Additional Notes

- This is the most stable algorithm (and slowest for n >> p).
"""
struct QR_GLS{QR} <: Algorithm
    decomp::QR
end
function qr_coef(decomp)
    R = decomp.R
    p = size(R, 1) - 1
    @views R[1:p, 1:p] \ R[1:p, end]
end

function QR_GLS(x, y, ::UniformScaling = I)
    decomp = qr([x y])
    QR_GLS(decomp, qr_coef(decomp))
end

function QR_GLS(x, y, v_invsqrt)
    decomp = qr(v_invsqrt * [x y])
    QR_GLS(decomp, qr_coef(decomp))
end


end
