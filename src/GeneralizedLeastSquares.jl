module GeneralizedLeastSquares

using LinearAlgebra
using LinearAlgebra: BlasFloat, BlasComplex
using SweepOperator: sweep!
using Statistics: mean

#-----------------------------------------------------------------------------# Algorithm
abstract type Algorithm end

function Base.show(io::IO, g::T) where {T<:Algorithm}
    nm = replace(string(T), "GeneralizedLeastSquares." => "")
    print(IOContext(io, :compact=>true), "$nm with coef: ", coef(g)')
end

#-----------------------------------------------------------------------------# SweepGLS
struct SweepGLS{T} <: Algorithm
    a::Matrix{T}
    s::Matrix{T}
end
coef(model::SweepGLS) = model.s[1:end-1, end]

function SweepGLS(x::StridedMatrix{T}, y::StridedVector{T}, ::UniformScaling=I) where {T<:Union{BlasFloat, BlasComplex}}
    n, p = size(x)
    a = zeros(T, p + 1, p + 1)
    @inbounds @views begin
        BLAS.syrk!('U', 'T', T(1/n), x, zero(T), a[1:p, 1:p]) # x'x
        BLAS.gemv!('T', T(1/n), x, y, zero(T), a[1:p, end])   # x'y
        a[end] = mean(abs2, y)                                # y'y
    end
    SweepGLS(a, sweep!(copy(a), 1:p))
end

# function SweepGLS(x::StridedMatrix{T}, y::StridedVector{T}, vinv) where {T<:Union{BlasFloat, BlasComplex}}
#     n, p = size(x)
#     a = zeros(T, p + 1, p + 1)
#     a[1:p, 1:p] = x'vinvsqrt

#     @views mul!(a[1:p,1:p], x')
#     a[1:p, 1:p] =
#     SweepGLS(a, sweep!(copy(a), 1:p))
# end

function SweepGLS(x, y, v_inv)
    n, p = size(x)
    a = zeros(promote_type(eltype(x), eltype(y), eltype(v_inv)), p + 1, p + 1)
    @inbounds begin
        xtv = x'v_inv
        a[1:p, 1:p] .= xtv * x ./ n
        a[1:p, end] .= xtv * y ./ n
        a[end] = y' * v_inv * y
    end
    SweepGLS(a, sweep!(copy(a), 1:p))
end


# #-----------------------------------------------------------------------------# _a
# # Create the upper triangle of the "augmented matrix:"
# #
# #  x'x (p × p)    x'y (p × 1)
# #                 y'y (1 × 1)
# function _a(y::StridedVector{T}, x::StridedMatrix{S}) where {T<:LinearAlgebra.BlasFloat, S<:LinearAlgebra.BlasFloat}
#     n, p = size(x)
#     a = zeros(T, p+1, p+1)
#     @views BLAS.syrk!('U', 'T', one(T), x, zero(T), a[1:p, 1:p]) # x'x
#     @views BLAS.gemv!('T', one(T), x, y, zero(T), a[1:p, end])   # x'y
#     a[end] = dot(y, y)                             # y'y
#     a
# end

# # fallback
# function _a(y::AbstractVector, x::AbstractMatrix)
#     n, p = size(x)
#     T = promote_type(eltype(y), eltype(x))
#     a = zeros(T, p+1, p+1)
#     @views mul!(a[1:p, 1:p], x', x)
#     @views mul!(a[1:p, end], x', y)
#     a[end] = dot(y, y)
#     return a
# end


# #-----------------------------------------------------------------------------# gls_vinvsqrt
# function gls_vinvsqrt(y::StridedVector{T}, x::StridedMatrix{T}, ::UniformScaling) where {T<:LinearAlgebra}


# function gls_vinvsqrt(y::StridedVector{T}, x::StridedMatrix{T}, vinvsqrt) where {T<:LinearAlgebra.BlasFloat}
#     n, p = size(x)
#     s = zeros(T, p+1, p+1)
#     α = T(1 / sqrt(n))
#     @views begin
#         mul!(s[1:p, 1:p], x', vinvsqrt, α, zero(T))   # X'sqrt(W) / sqrt(n)
#         mul!(s[1:p, end], vinvsqrt, y, α, zero(T))  # X'sqrt(W) / sqrt(n)
#         a = _a(s[1:p, end], s[1:p,1:p])
#     end
#     copy!(s, a)
#     sweep!(s, 1:p)
#     GLS(a, s)
# end

end
