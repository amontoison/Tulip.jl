mutable struct LN <: AbstractKKTSolver{Float64}
    m::Int  # Number of rows
    n::Int  # Number of columns

    # Problem data
    A::SparseMatrixCSC{Float64, Int}
    θ::Vector{Float64}
    regP::Vector{Float64}  # primal regularization
    regD::Vector{Float64}  # dual regularization
    variant::Bool

    function LN(A::SparseMatrixCSC{Float64}; variant::Bool=false)

        m, n = size(A)
        θ = ones(n)
        regP = ones(n)
        regD = ones(m)

        return new(m, n, A, θ, regP, regD, variant)

        return kkt
    end
end

setup(::Type{LN}, A) = LN(A)
backend(::LN) = "LN"
linear_system(::LN) = "least-norm formulation"

"""
    update!(kkt, θ, regP, regD)
"""
function update!(kkt::LN, θ::AbstractVector{Float64}, regP::AbstractVector{Float64}, regD::AbstractVector{Float64})

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
function solve!(dx::Vector{Float64}, dy::Vector{Float64}, kkt::LN, ξp::Vector{Float64}, ξd::Vector{Float64})

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
    z = zeros(Float64, n+m)
    if !variant
        F = qr([M̅⁻¹*Aᵀ; N̅])
        L = LowerTriangular(F.R')
        rhs = ξp + A * (M⁻¹ * ξd)
        permute!(rhs, F.pcol)
        z[1:m] = L \ rhs
        Δt = F.Q * z
        invpermute!(Δt, F.prow)
        Δy = N̅⁻¹ * Δt[n+1:m+n]
        Δx = M⁻¹ * (Aᵀ * Δy - ξd)
    else
        F = qr([N̅⁻¹*A; M̅])
        L = LowerTriangular(F.R')
        rhs = Aᵀ * (N⁻¹ * ξp) - ξd
        permute!(rhs, F.pcol)
        z[1:n] = L \ rhs
        Δs = F.Q * z
        invpermute!(Δs, F.prow)
        Δx = M̅⁻¹ * Δs[m+1:m+n]
        Δy = N⁻¹ * (ξp - A * Δx)
    end
    dx .= Δx
    dy .= Δy

    return nothing
end
