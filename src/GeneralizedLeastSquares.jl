module GeneralizedLeastSquares

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasComplex, checksquare
using SweepOperator: sweep!
using Statistics: mean

#-----------------------------------------------------------------------------# Notation
# Objective:    argmin_β (y - Xβ)' W (y - Xβ)
# Normal Equation:    (X'WX) β = X'Wy

#-----------------------------------------------------------------------------# LinRegAlgorithm
# Object that calculates β and can (on command) calculate the rest of the interface functions.
abstract type LinRegAlgorithm end

function Base.show(io::IO, g::T) where {T<:LinRegAlgorithm}
    nm = replace(string(T), "GeneralizedLeastSquares." => "")
    print(IOContext(io, :compact=>true), "$nm with coef: ", coef(g)')
end

# Interface
coef(model::LinRegAlgorithm) = model.β
predict(model::LinRegAlgorithm, x::AbstractMatrix) = x * model.β
predict(model::LinRegAlgorithm, x::AbstractVector) = dot(x, model.β)


#-----------------------------------------------------------------------------# Matrix Operations
const BlasVector{T} = StridedVector{T} where {T<:LinearAlgebra.BlasFloat}
const BlasMatrix{T} = StridedMatrix{T} where {T<:LinearAlgebra.BlasFloat}
const AbstractW = Union{<:BlasMatrix, <:UniformScaling}

# Required components:
# - cholesky:       X'WX and X'Wy
# - QR:             thin QR of X
# - sweep:          [X'WX X'Wy]
# - BunchKaufman:   X'WX and X'Wy
#-----------------------------------------------------------------------------# LinRegQR
mutable struct LinRegQR{T, Y<:BlasVector{T}} <: LinRegAlgorithm
    y::Y
    qrx::LinearAlgebra.QRCompactWY{T, Matrix{T}}
    β::Vector{T}
end
function LinRegQR(x, y)
    qrx = qr(x)
    β = qrx \ y
    LinRegQR(y, qrx, β)
end
LinRegQR(x, y, v) = (v2 = sqrt(v); LinRegQR(v2*x, v2*y))

function update!(model::LinRegQR, x::AbstractMatrix, y::AbstractVector)
    resize!(model.y, length(y))
    copy!(model.y, y)
    model.qrx = qr(x)
    model.β = model.qrx \ model.y
    model
end
function update!(model::LinRegQR{T}, x, y, w) where {T}
    w2 = sqrt(w)
    update!(model, w2 * x, w2 * y)
    model
end

#-----------------------------------------------------------------------------# LinRegCholesky
struct LinRegCholesky{T} <: LinRegAlgorithm
    A::BlasMatrix{T}    # X'WX
    b::BlasVector{T}    # X'Wy
    β::Vector{T}
    LinRegCholesky{T}(p::Integer) where {T} = new{T}(zeros(T, p, p), zeros(T, p), zeros(T, p))
end
LinRegCholesky(x, args...) = update!(LinRegCholesky{eltype(x)}(size(x,2)), x, args...)

function update!(model::LinRegCholesky{T}, x::AbstractMatrix, y::AbstractVector, w; buffer=zeros(T, size(x)...)) where {T}
    A, b = model.A, model.b
    mul!(buffer, w, x)
    α = T(1 / length(y))
    mul!(A, buffer', x, α, zero(T))
    mul!(b, buffer', y, α, zero(T))
    model.β[:] = cholesky(Hermitian(A)) \ b
    model
end
function update!(model::LinRegCholesky{T}, x::BlasMatrix{T}, y::BlasVector{T}) where {T}
    A, b = model.A, model.b
    α = T(1 / length(y))
    BLAS.syrk!('U', 'T', α, x, zero(T), A)
    BLAS.gemv!('T', α, x, y, zero(T), b)
    model.β[:] = cholesky(Hermitian(A)) \ b
    model
end

#-----------------------------------------------------------------------------# LinRegBunchKaufman
struct LinRegBunchKaufman{T} <: LinRegAlgorithm
    A::BlasMatrix{T}    # X'WX
    b::BlasVector{T}    # X'Wy
    β::Vector{T}
    LinRegBunchKaufman{T}(p::Integer) where {T} = new{T}(zeros(T, p, p), zeros(T, p), zeros(T, p))
end
LinRegBunchKaufman(x, args...) = update!(LinRegBunchKaufman{eltype(x)}(size(x,2)), x, args...)

function update!(model::LinRegBunchKaufman{T}, x::AbstractMatrix, y::AbstractVector, w; buffer=zeros(T, size(x)...)) where {T}
    A, b = model.A, model.b
    mul!(buffer, w, x)
    α = T(1 / length(y))
    mul!(A, buffer', x, α, zero(T))
    mul!(b, buffer', y, α, zero(T))
    model.β[:] = bunchkaufman(Hermitian(A)) \ b
    model
end
function update!(model::LinRegBunchKaufman{T}, x::BlasMatrix{T}, y::BlasVector{T}) where {T}
    A, b = model.A, model.b
    α = T(1 / length(y))
    BLAS.syrk!('U', 'T', α, x, zero(T), A)
    BLAS.gemv!('T', α, x, y, zero(T), b)
    model.β[:] = bunchkaufman(Hermitian(A)) \ b
    model
end

#-----------------------------------------------------------------------------# LinRegSweep
struct LinRegSweep{T} <: LinRegAlgorithm
    A::BlasMatrix{T}    # [X'WX X'Wy]
    β::Vector{T}
    LinRegSweep{T}(p::Integer) where {T} = new{T}(zeros(T, p + 1, p + 1), zeros(T, p))
end
LinRegSweep(x, args...) = update!(LinRegSweep{eltype(x)}(size(x,2)), x, args...)

function update!(model::LinRegSweep{T}, x::AbstractMatrix, y::AbstractVector, w; buffer=zeros(T, size(x)...)) where {T}
    A = model.A
    p = size(A, 1) - 1
    mul!(buffer, w, x)
    α = T(1 / length(y))
    @views begin
        mul!(A[1:p, 1:p], buffer', x, α, zero(T))
        mul!(A[1:p, end], buffer', y, α, zero(T))
        sweep!(A, 1:p)
        copy!(model.β, A[1:p, end])
    end
    model
end
function update!(model::LinRegSweep{T}, x::AbstractMatrix, y::AbstractVector) where {T}
    A = model.A
    p = size(A, 1) - 1
    α = T(1 / length(y))
    @views begin
        BLAS.syrk!('U', 'T', α, x, zero(T), A[1:p, 1:p])
        BLAS.gemv!('T', α, x, y, zero(T), A[1:p, end])
        sweep!(A, 1:p)
        copy!(model.β, A[1:p, end])
    end
    model
end

end
