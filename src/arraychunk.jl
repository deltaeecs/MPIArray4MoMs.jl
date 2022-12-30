"""
Array chunk with custom indices.
```
data::Array{T, N}
indices::NTuple{N, Union{UnitRange{Int}, Vector{Int}}}
```
"""
struct ArrayChunk{T, N} <:AbstractArray{T, N}
    data::Array{T, N}
    indices::NTuple{N, Union{UnitRange{Int}, Vector{Int}}}
end


function ArrayChunk(T, indices::Vararg{Union{UnitRange{Int}, Vector{Int}}, N}) where {N}

    size =  map(length, indices)
    data =  zeros(T, size...)

    ArrayChunk{T, N}(data, indices)
end

ArrayChunk(data::Array{T, N}, indices) where {T,N} = ArrayChunk{T, N}(data, indices)

import Base:size, show, display, eltype, length, fill!, getindex, setindex!, sum

size(A::T) where{T<:ArrayChunk}  = size(A.data)
size(A::T, d::Integer) where{T<:ArrayChunk}  = size(A.data, d)
eltype(::ArrayChunk{T, N}) where{T, N} = T
length(A::T) where{T<:ArrayChunk} = prod(size(A))
fill!(A::T, x) where{T<:ArrayChunk} = fill!(A.data, x)
sum(A::T) where{T<:ArrayChunk} = sum(A.data)

"""
overload getindex
"""
function getindex(A::ArrayChunk{T, N}, idcs::Vararg{I, N}) where{T, N, I<:Integer}
    
    ptrs    =   map(searchsorted, A.indices, idcs)
    any(isempty, ptrs) && return zero(eltype(A.data))
    idcslc  =   map(first, ptrs)
    return A.data[idcslc...]

end

"""
overload setindex!
"""
function setindex!(A::ArrayChunk{T, N}, x, idcs::Vararg{I, N}) where{T, N, I<:Integer}

    ptrs    =   map(searchsorted, A.indices, idcs)
    any(isempty, ptrs) && begin @warn "$idcs not in indices of array!"; end
    idcslc  =   map(first, ptrs)
    setindex!(A.data, x, idcs...)
    return nothing
end
