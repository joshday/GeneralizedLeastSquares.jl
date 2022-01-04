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
    matrix::Matrix{T}
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
    SweepGLS(sweep!(a, 1:p))
end

function SweepGLS(x, y, v_inv)
    n, p = size(x)
    a = zeros(promote_type(eltype(x), eltype(y), eltype(v_inv)), p + 1, p + 1)
    @inbounds @views begin
        xtv = x'v_inv
        a[1:p, 1:p] = xtv * x ./ n
        a[1:p, end] = xtv * y ./ n
        a[end] = y' * v_inv * y / n
    end
    SweepGLS(sweep!(a, 1:p))
end


end
