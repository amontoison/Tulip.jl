mutable struct LS <: AbstractKKTSolver{Float64}
    m::Int  # Number of rows
    n::Int  # Number of columns

    # Problem data
    A::SparseMatrixCSC{Float64, Int}
    θ::Vector{Float64}
    regP::Vector{Float64}  # primal regularization
    regD::Vector{Float64}  # dual regularization
    variant::Bool

    function LS(A::SparseMatrixCSC{Float64}; variant::Bool=false)

        m, n = size(A)
        θ = ones(n)
        regP = ones(n)
        regD = ones(m)

        return new(m, n, A, θ, regP, regD, variant)

        return kkt
    end
end

setup(::Type{LS}, A) = LS(A)
backend(::LS) = "LS"
linear_system(::LS) = "least-squares formulation"

"""
    update!(kkt, θ, regP, regD)
"""
function update!(kkt::LS, θ::AbstractVector{Float64}, regP::AbstractVector{Float64}, regD::AbstractVector{Float64})

    # Sanity checks
    length(θ)  == kkt.n || throw(DimensionMismatch(
        "θ has length $(length(θ)) but linear solver is for n=$(kkt.n)."
    ))
    length(regP) == kkt.n || throw(DimensionMismatch(
        "regP has length $(length(regP)) but linear solver has n=$(kkt.n)"
    ))
    length(regD) == kkt.m || throw(DimensionMismatch(
        "regD has length $(length(regD)) but linear solver has m=$(kkt.m)"
    ))

    # Update diagonal scaling
    kkt.θ .= θ

    # Update regularizers
    kkt.regP .= regP
    kkt.regD .= regD

    return nothing
end

"""
    solve!(dx, dy, kkt, ξp, ξd)
"""
function solve!(dx::Vector{Float64}, dy::Vector{Float64}, kkt::LS, ξp::Vector{Float64}, ξd::Vector{Float64})

    m, n = kkt.m, kkt.n
    A = kkt.A
    Aᵀ = kkt.A'
    M = Diagonal(kkt.θ + kkt.regP)
    N = Diagonal(kkt.regD)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    M̅ = Diagonal(sqrt.(M.diag))
    N̅ = Diagonal(sqrt.(N.diag))
    M̅⁻¹ = inv(M̅)
    N̅⁻¹ = inv(N̅)
    if !kkt.variant
        F = qr([M̅⁻¹*Aᵀ; N̅])
        rhs = [M̅⁻¹*ξd; N̅⁻¹*ξp]
        permute!(rhs, F.prow)
        z = F.Q' * rhs
        Δy = UpperTriangular(F.R) \ view(z, 1:m)
        invpermute!(Δy, F.pcol)
        Δx = M⁻¹ * (Aᵀ * Δy - ξd)
    else
        F = qr([N̅⁻¹*A; M̅])
        rhs = [N̅⁻¹*ξp; -M̅⁻¹*ξd]
        permute!(rhs, F.prow)
        z = F.Q' * rhs
        Δx = UpperTriangular(F.R) \ view(z, 1:n)
        invpermute!(Δx, F.pcol)
        Δy = N⁻¹ * (ξp - A * Δx)
    end
    dx .= Δx
    dy .= Δy

    return nothing
end
