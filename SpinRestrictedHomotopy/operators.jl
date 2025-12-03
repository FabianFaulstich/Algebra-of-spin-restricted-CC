using HomotopyContinuation
using Combinatorics
using LinearAlgebra

## Helper function to run code with limited threading
# Useful when Julia is started with many threads but specific operations
# perform better with fewer threads (e.g., memory-intensive operations)
function with_limited_threads(f, num_threads::Int=1)
    old_blas = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(num_threads)
        return f()
    finally
        BLAS.set_num_threads(old_blas)
    end
end

## Helper function to run code with GC disabled
# Useful for avoiding GC overhead with many threads
# GC will run automatically after the function completes if needed
function with_gc_disabled(f)
    gc_was_enabled = GC.enable(false)
    try
        return f()
    finally
        GC.enable(gc_was_enabled)
        # Allow GC to run if it needs to
        GC.safepoint()
    end
end

## Combined helper: disable GC and limit threads
# Optimizes performance when running with many threads (e.g., 144)
# by avoiding GC thrashing and BLAS thread contention
function with_optimized_allocation(f, num_threads::Int=1)
    old_blas = BLAS.get_num_threads()
    gc_was_enabled = GC.enable(false)
    try
        BLAS.set_num_threads(num_threads)
        return f()
    finally
        BLAS.set_num_threads(old_blas)
        GC.enable(gc_was_enabled)
        GC.safepoint()
    end
end

## Structure defining the creation and annihilation operators

struct Op
    index::Int
    creation::Bool
end

Op(index::Int, creation::Bool, spin::Bool) = Op(spin ? 2*index : 2*index - 1, creation)

# Base.show(io::IO, op::Op) = print(io, op.creation ? "a$(op.index)†" : "a$(op.index)")
# Base.show(io::IO, ::MIME"text/latex", op::Op) =
#     print(io, op.creation ? "a_{$(op.index)}^{\\dagger}" : "a_{$(op.index)}")
# Base.show(io::IO, ::MIME"text/html", op::Op) =
#     print(io, op.creation ? "\$a_{$(op.index)}^{\\dagger}\$" : "\$a_{$(op.index)}\$")

Base.isless(a::Op, b::Op) = (a.creation && !b.creation) || (a.creation && b.creation && a.index > b.index) || (!a.creation && !b.creation && a.index < b.index) 


## A word in the the Fermi Dirac algebra with a constant that can be a Number or a Homotopy Continuation Expression

struct FDword{T}
    monomial::Union{Vector{Op}, Nothing}
    constant::T
end

FDword(mon::Union{Vector{Op}, Nothing}) = FDword(mon, 1)
FDword() = FDword(nothing, 1)
FDword{T}(mon::Union{Vector{Op}, Nothing}) where T = FDword{T}(mon, one(T))
FDword{T}() where T = FDword{T}(nothing, one(T))

function Base.:*(a::FDword{T1}, b::FDword{T2}) where {T1,T2}
    # Combine monomials, handling nothing cases properly
    if isnothing(a.monomial) && isnothing(b.monomial)
        monomial = nothing
    elseif isnothing(a.monomial)
        monomial = b.monomial
    elseif isnothing(b.monomial)
        monomial = a.monomial
    else
        monomial = [a.monomial; b.monomial]
    end

    constant = a.constant * b.constant
    return FDword(monomial, constant)
end

function Base.:*(a::T1, b::FDword{T2}) where {T1<:Union{Expression, Variable, Number}, T2}
    constant = a * b.constant
    return FDword(b.monomial, constant)
end

# Reverse operations for FDword * scalar
Base.:*(a::FDword{T1}, b::T2) where {T1, T2<:Union{Expression, Variable, Number}} = FDword(a.monomial, a.constant * b)


# Promotion rules: Op automatically promotes to FDword{Int}
Base.convert(::Type{FDword{T}}, op::Op) where T = FDword([op], one(T))
Base.convert(::Type{FDword}, op::Op) = FDword([op], 1)
Base.promote_rule(::Type{Op}, ::Type{FDword{T}}) where T = FDword{T}
Base.promote_rule(::Type{Op}, ::Type{Op}) = FDword{Int}

# Tell Julia to use promotion for these operations
Base.:*(a::Op, b::Op) = convert(FDword, a) * convert(FDword, b)
Base.:+(a::Op, b::Op) = convert(FDword, a) + convert(FDword, b)
Base.:*(a::Op, b::FDword{T}) where T = convert(FDword{T}, a) * b
Base.:*(a::FDword{T}, b::Op) where T = a * convert(FDword{T}, b)
Base.:+(a::Op, b::FDword{T}) where T = convert(FDword{T}, a) + b
Base.:+(a::FDword{T}, b::Op) where T = a + convert(FDword{T}, b)

# Op * scalar  
Base.:*(a::Op, b::T) where T<:Union{Expression, Variable, Number} = FDword([a], b)
Base.:*(a::T, b::Op) where T<:Union{Expression, Variable, Number} = FDword([b], a)

function find_pairings(as::Vector{Int}, adags::Vector{Int}, str::Vector{Op})::Vector{Vector{Vector{Int}}}
    length(as) == length(adags) || return []
    (first(adags) < first(as) || last(adags) < last(as)) && return []
    length(as) == 1 && return str[first(as)].index == str[first(adags)].index ? [[[first(as), first(adags)]]] : []

    a, rest_as... = as

    is = findall(ad -> str[ad].index == str[a].index ,adags)
    isnothing(is) && return []

    pairs = []
    for i in is
        c, rest_adags = adags[i], adags[1:end.!=i]
        sub_contractions = find_pairings(rest_as, rest_adags, str)
        append!(pairs, [[[a, c]];sub] for sub in sub_contractions)
    end
    return pairs
end

function wick_sign(pairing::Vector{Vector{Int}})
   # given two pairs [a, b], and [c, d].
   # How to decide whether they cross?
   return prod([(a < c < b < d || c < a < d < b) ? - 1 : 1 for ((a,b), (c,d)) in combinations(pairing, 2)])
end

function wick_contraction(str::Vector{Op})
    as = findall(op -> !op.creation, str)
    adags = findall(op -> op.creation, str)
    pairs = find_pairings(as, adags, str);
    return sum([wick_sign(pair) for pair in pairs])
end

function wick_contraction(str::FDword)
    return isnothing(str.monomial) ? 1 : str.constant*wick_contraction(str.monomial)
end

# function find_pairing(as::Vector{Int}, adags::Vector{Int}, str::Vector{Op})::Union{Vector{Vector{Int}}, Nothing}
#     length(as) == length(adags) || return nothing
#     (first(adags) < first(as) || last(adags) < last(as)) && return nothing
#     length(as) == 1 && return str[first(as)].index == str[first(adags)].index ?
#     [[first(as), first(adags)]] : nothing

#     a, rest_as... = as

#     i = findfirst(ad -> str[ad].index == str[a].index ,adags)
#     isnothing(i) && return nothing

#     c, rest_adags = adags[i], adags[1:end.!=i]
#     sub_contraction = find_pairing(rest_as, rest_adags, str)
#     return isnothing(sub_contraction) ? nothing : [[[a, c]];sub_contraction] 
# end

# function wick_sign(pairing::Vector{Vector{Int}})
#    # given two pairs [a, b], and [c, d].
#    # How to decide whether they cross?
#    return prod([(a < c < b < d || c < a < d < b) ? - 1 : 1 for ((a,b), (c,d)) in combinations(pairing, 2)])
# end

# function wick_contraction(str::Vector{Op})
#     as = findall(op -> !op.creation, str)
#     adags = findall(op -> op.creation, str)
#     pair = find_pairing(as, adags, str);
#     return isnothing(pair) ? 0 : wick_sign(pair)
# end

# function wick_contraction(str::FDword)
#     return isnothing(str.monomial) ? 1 : expand(str.constant*wick_contraction(str.monomial))
# end


# Taken from Permutation.jl
function perm_sign(p::Vector)
    n = length(p)
    result = 0
    todo = trues(n)
    while any(todo)
        k = findfirst(todo)
        todo[k] = false
        result += 1 # increment element count
        j = p[k]
        while j != k
            result += 1 # increment element count
            todo[j] = false
            j = p[j]
        end
        result += 1 # increment cycle count
    end
    return isodd(result) ? -1 : 1
end

function o(str::FDword)
    isnothing(str.monomial) && return str
    op = str.monomial;
    perm = sortperm(op);
    return FDword(op[perm], perm_sign(perm)*str.constant)
end

function isext(str::FDword)
    return isnothing(str.monomial) || length(unique([op.index for op in unique(str.monomial)])) == length(unique(str.monomial))
end

function iszero(str::FDword)
    str.constant == 0 && return true
    isnothing(str.monomial) && return false

    ops = str.monomial
    for i in 1:length(ops)
        op = ops[i];
        sim_op = findfirst(o -> o.index == op.index , ops[i + 1:end]);
        (!(isnothing(sim_op)) && op.creation == ops[i + sim_op].creation) && return true
    end

    return false
end


## A structure on general element in the Fermi Dirac algebra

struct FD{T}
    sum::Vector{FDword{T}}

    function FD{T}(sum::Vector{FDword{T}}) where T
        # Filter out words that:
        # 1. Have Nothing as monomial and 0 as constant
        # 2. Have duplicate operators (same operator appears twice)
        filtered = filter(w -> !iszero(w), sum)
        new{T}(filtered)
    end
end

# Convenience constructor that infers type from input
FD(sum::Vector{FDword{T}}) where T = FD{T}(sum)

# Constructor for mixed types - promotes to common type
function FD(sum::Vector{<:FDword})
    # Find the promoted type of all constants
    types = [typeof(w.constant) for w in sum]
    T = promote_type(types...)
    # Convert all FDwords to use the promoted type
    promoted_sum = [FDword(w.monomial, convert(T, w.constant)) for w in sum]
    return FD{T}(promoted_sum)
end

# Base.show(io::IO, vec::FD) = print(io, join(vec.sum, " + " ))

function simplify(fd::FD{T}) where T
    # Pre-size dictionary to avoid rehashing (major source of contention)
    dict = Dict{Union{Vector{Op}, Nothing}, T}()
    sizehint!(dict, length(fd.sum))

    for word in fd.sum
        w = isext(word) ? o(word) : word

        # Use haskey instead of 'in keys()' - much faster
        if haskey(dict, w.monomial)
            dict[w.monomial] += w.constant
        else
            dict[w.monomial] = w.constant
        end
    end

    # Build result without expand() - let coefficients stay as-is
    result_words = FDword{T}[]
    sizehint!(result_words, length(dict))

    for (mon, coef) in dict
        if coef != 0
            push!(result_words, FDword(mon, coef))
        end
    end

    return FD(result_words)
end

## Force expand on an FD object (call manually if needed for display)
function expand_all(fd::FD{T}) where T
    if T == Expression
        return FD([FDword(w.monomial, expand(w.constant)) for w in fd.sum])
    else
        return fd
    end
end

Base.zero(::FD{T}) where T = FD(FDword{T}[])
Base.one(::FD{T}) where T = FD([FDword{T}()])

# Type versions needed for matrix operations like det
Base.zero(::Type{FD{T}}) where T = FD(FDword{T}[])
Base.one(::Type{FD{T}}) where T = FD([FDword{T}()])

# Copy constructor needed for oneunit
FD{T}(x::FD{T}) where T = x

# oneunit for matrix operations
Base.oneunit(::Type{FD{T}}) where T = one(FD{T})

function Base.:+(a::FD{T1}, b::FD{T2}) where {T1,T2}
    if T1 == T2
        return simplify(FD([a.sum;b.sum]))
    else
        T = promote_type(T1, T2)
        a_promoted = [FDword(w.monomial, convert(T, w.constant)) for w in a.sum]
        b_promoted = [FDword(w.monomial, convert(T, w.constant)) for w in b.sum]
        return simplify(FD{T}([a_promoted; b_promoted]))
    end
end

function Base.:+(a::T1, b::FD{T2}) where {T1<:Union{Expression, Variable, Number}, T2}
    T = promote_type(T1, T2)
    a_word = FDword(nothing, convert(T, a))
    b_promoted = [FDword(w.monomial, convert(T, w.constant)) for w in b.sum]
    return simplify(FD{T}([a_word; b_promoted]))
end

function Base.:-(a::FD{T1}, b::FD{T2}) where {T1,T2}
    if T1 == T2
        return simplify(FD([a.sum;((-1)*b).sum]))
    else
        return a + ((-1)*b)
    end
end

function Base.:*(a::FD{T1}, b::FD{T2}) where {T1,T2}
    products = [aword*bword for aword in a.sum for bword in b.sum]
    return simplify(FD(products))
end

# FD * FDword and FDword * FD
function Base.:*(a::FD{T1}, b::FDword{T2}) where {T1,T2}
    products = [aword*b for aword in a.sum]
    return simplify(FD(products))
end

function Base.:*(a::FDword{T1}, b::FD{T2}) where {T1,T2}
    products = [a*bword for bword in b.sum]
    return simplify(FD(products))
end

function Base.:*(a::T1, b::FD{T2}) where {T1<:Union{Expression, Variable, Number}, T2}
    return simplify(FD([a*bword for bword in b.sum]))
end

# Reverse operations for FD
function Base.:+(a::FD{T1}, b::T2) where {T1, T2<:Union{Expression, Variable, Number}}
    T = promote_type(T1, T2)
    a_promoted = [FDword(w.monomial, convert(T, w.constant)) for w in a.sum]
    b_word = FDword(nothing, convert(T, b))
    return simplify(FD{T}([a_promoted; b_word]))
end
Base.:*(a::FD{T1}, b::T2) where {T1, T2<:Union{Expression, Variable, Number}} = simplify(FD([aword*b for aword in a.sum]))

# Subtraction with constants
function Base.:-(a::FD{T1}, b::T2) where {T1, T2<:Union{Expression, Variable, Number}}
    T = promote_type(T1, T2)
    a_promoted = [FDword(w.monomial, convert(T, w.constant)) for w in a.sum]
    b_word = FDword(nothing, convert(T, -b))
    return simplify(FD{T}([a_promoted; b_word]))
end

function Base.:-(a::T1, b::FD{T2}) where {T1<:Union{Expression, Variable, Number}, T2}
    T = promote_type(T1, T2)
    a_word = FDword(nothing, convert(T, a))
    b_negated = [FDword(w.monomial, convert(T, -w.constant)) for w in b.sum]
    return simplify(FD{T}([a_word; b_negated]))
end

# Unary minus for FD
Base.:-(a::FD{T}) where T = FD([FDword(w.monomial, -w.constant) for w in a.sum])

# Conversion and promotion rules: FDword automatically promotes to FD
Base.convert(::Type{FD{T}}, fw::FDword{T}) where T = FD([fw])
Base.convert(::Type{FD{T}}, fw::FDword{S}) where {T,S} = FD([FDword(fw.monomial, convert(T, fw.constant))])
Base.convert(::Type{FD}, fw::FDword{T}) where T = FD([fw])
Base.promote_rule(::Type{FDword{T}}, ::Type{FD{S}}) where {T,S} = FD{promote_type(T,S)}
Base.promote_rule(::Type{FDword{T}}, ::Type{FDword{S}}) where {T,S} = FD{promote_type(T,S)}

# FDword + FDword creates FD (handles mixed types)
function Base.:+(a::FDword{T1}, b::FDword{T2}) where {T1,T2}
    if T1 == T2
        return FD([a,b])
    else
        T = promote_type(T1, T2)
        a_promoted = FDword(a.monomial, convert(T, a.constant))
        b_promoted = FDword(b.monomial, convert(T, b.constant))
        return FD([a_promoted, b_promoted])
    end
end

# FDword + FD handles mixed types
function Base.:+(a::FDword{T1}, b::FD{T2}) where {T1,T2}
    if T1 == T2
        return simplify(FD{T1}([a; b.sum]))
    else
        T = promote_type(T1, T2)
        a_conv = FDword(a.monomial, convert(T, a.constant))
        b_conv = FDword{T}[FDword(w.monomial, convert(T, w.constant)) for w in b.sum]
        return simplify(FD{T}([a_conv; b_conv]))
    end
end

function Base.:+(a::FD{T1}, b::FDword{T2}) where {T1,T2}
    if T1 == T2
        return simplify(FD{T1}([a.sum; b]))
    else
        T = promote_type(T1, T2)
        a_conv = FDword{T}[FDword(w.monomial, convert(T, w.constant)) for w in a.sum]
        b_conv = FDword(b.monomial, convert(T, b.constant))
        return simplify(FD{T}([a_conv; b_conv]))
    end
end

# Reverse operations for better type compatibility  
Base.:+(a::FDword{T1}, b::T2) where {T1, T2<:Union{Expression, Variable, Number}} = FD([a, FDword(nothing, b)])

# Op + FD operations (must be after FD is defined)
Base.:+(a::Op, b::FD{T}) where T = convert(FDword{T}, a) + b
Base.:+(a::FD{T}, b::Op) where T = a + convert(FDword{T}, b)

Base.:^(a::FD{T}, n::Integer) where T = n == 0 ? FD([FDword(nothing, one(T))]) : a*a^(n - 1)

function com(a::FD{T1}, b::FD{T2}) where {T1,T2}
    return a*b - b*a
end

function comit(a::FD{T1}, b::FD{T2}, n::Int64) where {T1,T2}
    return n == 0 ? a : com(comit(a,b,n - 1), b)
end

function matrixrep(fd::FD{T1}, bra_ops::Vector{<:FD}, ket_ops::Vector{<:FD}; threaded::Bool=false) where T1
    nrows = length(bra_ops)
    ncols = length(ket_ops)

    # Handle empty FD case
    if isempty(fd.sum)
        return zeros(Int, nrows, ncols)
    end

    # If bra_ops and ket_ops are the same object, copy to avoid modifying both
    if bra_ops === ket_ops
        bra_ops = copy(bra_ops)
    end

    # Determine the promoted type from all FD objects
    # Extract the type parameter T from FD{T}
    get_fd_type(::FD{T}) where T = T

    bra_types = [get_fd_type(bra) for bra in bra_ops if !isempty(bra.sum)]
    ket_types = [get_fd_type(ket) for ket in ket_ops if !isempty(ket.sum)]
    all_types = [T1; bra_types; ket_types]
    T = isempty(all_types) ? T1 : promote_type(all_types...)

    # Save current BLAS thread count
    old_blas_threads = BLAS.get_num_threads()

    try
        # Always set BLAS to 1 thread during matrixrep to avoid contention
        # This is the key optimization for running with many Julia threads (e.g., 144)
        BLAS.set_num_threads(1)

        for j in 1:nrows
            ops = bra_ops[j]
            bra_ops[j] = FD([FDword([Op(o.index, !o.creation) for o in reverse(w.monomial)], w.constant) for w in ops.sum])
        end

        result = Matrix{T}(undef, nrows, ncols)

        # Compute matrix elements
        # threaded=false (default): Single-threaded - best for most cases with many threads
        # threaded=true: Use Julia threading - only beneficial for very large matrices
        if threaded
            Threads.@threads for j in 1:ncols
                @inbounds for i in 1:nrows
                    long_str = bra_ops[i] * fd * ket_ops[j]
                    # Avoid allocating array in sum - accumulate directly
                    acc = zero(T)
                    for op in long_str.sum
                        acc += wick_contraction(op)
                    end
                    result[i,j] = acc
                end
            end
        else
            @inbounds for j in 1:ncols
                for i in 1:nrows
                    long_str = bra_ops[i] * fd * ket_ops[j]
                    # Avoid allocating array in sum - accumulate directly
                    acc = zero(T)
                    for op in long_str.sum
                        acc += wick_contraction(op)
                    end
                    result[i,j] = acc
                end
            end
        end

        return result
    finally
        # Restore original BLAS thread count
        BLAS.set_num_threads(old_blas_threads)
    end
end

function matrixrep(fd::FD{T}, rowbasis::Vector{Vector{Int64}}, colbasis::Vector{Vector{Int64}}; threaded::Bool=false) where T
    nrows = length(rowbasis)
    ncols = length(colbasis)

    # Handle empty FD case
    if isempty(fd.sum)
        return zeros(Int, nrows, ncols)
    end

    # Pre-compute bra operators for all row basis states
    # Do this with threading disabled to avoid allocation contention
    bra_ops = Vector{FD}(undef, nrows)
    for i in 1:nrows
        bra_ops[i] = FD([FDword([Op(idx, true) for idx in rowbasis[i]])])
    end

    # Pre-compute ket operators for all column basis states
    ket_ops = Vector{FD}(undef, ncols)
    for j in 1:ncols
        ket_ops[j] = FD([FDword([Op(idx, true) for idx in colbasis[j]])])
    end

    return matrixrep(fd, bra_ops, ket_ops; threaded)
end

function matrixrep(fd::FD{T}, basis::Vector{Vector{Int64}}; threaded::Bool=false) where T
    return matrixrep(fd, basis, basis; threaded=threaded)
end

# Custom det for FD matrices (non-commutative aware)
import LinearAlgebra: det
function det(A::Matrix{FD{T}}) where T
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    
    if n == 1
        return A[1,1]
    elseif n == 2
        # For 2x2: det = a11*a22 - a12*a21 (order matters!)
        return A[1,1]*A[2,2] - A[1,2]*A[2,1]
    else
        # For larger matrices, use cofactor expansion along first row
        result = zero(FD{T})
        for j in 1:n
            # Create minor matrix
            minor = A[[i for i in 1:n if i != 1], [k for k in 1:n if k != j]]
            cofactor = (iseven(j) ? 1 : -1) * det(minor)
            result = result + A[1,j] * cofactor
        end
        return result
    end
end