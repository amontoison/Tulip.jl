using LinearAlgebra.LAPACK

"""
    SparseLinearSolver{T}

"""
mutable struct DenseLinearSolver{T<:Real} <: TLPLinearSolver{T}
    m::Int  # Number of rows in A
    n::Int  # Number of columns in A

    # Problem data
    A::Matrix{T}
    θ::Vector{T}

    B::Matrix{T}  # place-holder to avoid allocating memory afterwards

    # Factorization
    S::Matrix{T}  # Normal equations matrix, that also holds the factorization

    # Constructor
    function DenseLinearSolver(A::Matrix{T}) where{T<:Real}
        m, n = size(A)
        θ = ones(T, n)

        # We just need to allocate memory here,
        # so no factorization yet
        S = zeros(T, m, m)

        return new{T}(m, n, A, θ, copy(A), S)
    end
end

TLPLinearSolver(A::Matrix{Tv}) where{Tv<:BlasReal} = DenseLinearSolver(A)

function update_linear_solver(
    ls::DenseLinearSolver{T},
    d::AbstractVector{T}
) where{T<:BlasReal}
    # Sanity checks
    length(d) == ls.n || throw(DimensionMismatch(
        "d has length $(length(d)) but linear solver is for n=$(ls.n)."
    ))

    ls.θ .= d

    # Re-compute normal equations matrix
    # There's no function that does S = A*D*A', so we cache a copy of A
    copyto!(ls.B, ls.A)
    rmul!(ls.B, Diagonal(sqrt.(ls.θ)))  # B = A * √D
    BLAS.syrk!('U', 'N', one(T), ls.B, zero(T), ls.S)
    
    # Cholesky factorization
    # TODO: regularize if needed
    LAPACK.potrf!('U', ls.S)
    
    return nothing
end


"""
    solve_augmented_system!(dx, dy, A, ls, θ, ξp, ξd)

Solve the augmented system, overwriting `dx, dy` with the result.
```
    -Θ*dx  + A'dy = ξd
     A*dx         = ξp
```

For dense `A` we first compute the normal equations system.

`A` and `θ` are given to perform iterative refinement if needed.

# Arguments
- `dx::Vector{Tv}, dy::Vector{Tv}`: Vectors of unknowns, modified in-place
- `ls::DenseLinearSolver{Tv}`: Linear solver for the augmented system
- `A::Matrix{Tv}`: Constraint matrix
- `θ::Vector{Tv}`: Diagonal scaling vector
- `ξp::Vector{Tv}, ξd::Vector{Tv}`: Right-hand-side vectors
"""
function solve_augmented_system!(
    dx::Vector{Tv}, dy::Vector{Tv},
    ls::DenseLinearSolver{Tv}, A::Matrix{Tv}, θ::Vector{Tv},
    ξp::Vector{Tv}, ξd::Vector{Tv}
) where{Tv<:BlasReal}
    m, n = size(A)
    
    # Set-up right-hand side
    ξp_ = ξp + ls.A * (ξd .* θ)

    # Solve augmented system
    copyto!(dy, ξp_)
    LAPACK.potrs!('U', ls.S, dy)  # potrs! over-writes the right-hand side

    # Recover dx
    # TODO: use more efficient mul! syntax
    dx .= (ls.A' * dy - ξd) .* θ

    # TODO: Iterative refinement
    return nothing
end