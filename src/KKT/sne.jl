mutable struct SNE <: AbstractKKTSolver{Float64}
    m::Int  # Number of rows
    n::Int  # Number of columns

    # Problem data
    A::SparseMatrixCSC{Float64, Int}
    θ::Vector{Float64}
    regP::Vector{Float64}  # primal regularization
    regD::Vector{Float64}  # dual regularization
    variant::Bool

    function SNE(A::SparseMatrixCSC{Float64}; variant::Bool=false)

        m, n = size(A)
        θ = ones(n)
        regP = ones(n)
        regD = ones(m)

        return new(m, n, A, θ, regP, regD, variant)

        return kkt
    end
end

setup(::Type{SNE}, A) = SNE(A)
backend(::SNE) = "SNE"
linear_system(::SNE) = "seminormal equations"

"""
    update!(kkt, θ, regP, regD)
"""
function update!(kkt::SNE, θ::AbstractVector{Float64}, regP::AbstractVector{Float64}, regD::AbstractVector{Float64})

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
function solve!(dx::Vector{Float64}, dy::Vector{Float64}, kkt::SNE, ξp::Vector{Float64}, ξd::Vector{Float64})

    m, n = kkt.m, kkt.n
    A = kkt.A
    Aᵀ = kkt.A'
    M = Diagonal(kkt.θ + kkt.regP)
    N = Diagonal(kkt.regD)
    M⁻¹ = inv(M)
    N⁻¹ = inv(N)
    M̅ = Diagonal(sqrt.(M.diag))
    N̅ = Diagonal(sqrt.(N.diag))
    if !variant
        M̅⁻¹ = inv(M̅)
        F = qr([M̅⁻¹*Aᵀ; N̅])  # F.Q * F.R = A[F.prow, F.pcol] = Prow * A * Pcol
        R = UpperTriangular(F.R)
        y = zeros(Float64, m)
        T⁻¹ = LinearOperator(Float64, m, m, true, true, v -> (y .= v ; permute!(y, F.pcol) ; ldiv!(R', y) ; ldiv!(R, y) ; invpermute!(y, F.pcol)))
        Δy = T⁻¹ * (ξp + A * M⁻¹ * ξd)
        Δx = M⁻¹ * (Aᵀ * Δy - ξd)
    else
        N̅⁻¹ = inv(N̅)
        F = qr([N̅⁻¹*A; M̅])  # F.Q * F.R = A[F.prow, F.pcol] = Prow * A * Pcol
        R = UpperTriangular(F.R)
        y = zeros(Float64, n)
        T⁻¹ = LinearOperator(Float64, n, n, true, true, v -> (y .= v ; permute!(y, F.pcol) ; ldiv!(R', y) ; ldiv!(R, y) ; invpermute!(y, F.pcol)))
        Δx = T⁻¹ * (Aᵀ * N⁻¹ * ξp - ξd)
        Δy = N⁻¹ * (ξp - A * Δx)
    end

    dx .= Δx
    dy .= Δy

    return nothing
end
