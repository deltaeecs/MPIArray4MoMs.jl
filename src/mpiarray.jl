
SubOrArray{T, N} = Union{Array{T, N}, SubArray{T,N,P,I,L}} where {T, N, P, I, L} 

"""
MPIArray used in MoM.

`data`, the data stored in local rank;

`indices`, indice of `data` in global Array;

`size`, global size of data;

`rank2indices`, a dict record all MPI ranks to their datas global indices,

`ghostdata`, the data stored or used in this rank,

`grank2ghostindices`, the data used but not stored in this rank, here provides the host rank and its indices in `ghostdata`,

`rrank2localindices`, the data host in this rank but used in other rank, here provides the remote rank and the indices used in `data`.
"""
mutable struct MPIArray{T, I, N}<:AbstractArray{T, N}
    data::SubOrArray{T, N}
    indices::I
    comm::MPI.Comm
    myrank::Int
    size::NTuple{N,Int}
    rank2indices::Dict{Int, I}
    ghostdata::Array{T, N}
    ghostindices::I
    grank2ghostindices::Dict{Int, I}
    rrank2localindices::Dict{Int, I}
end

MPIVector{T, I} = MPIArray{T, I, 1} where {T, I}
MPIMatrix{T, I} = MPIArray{T, I, 2} where {T, I}

Base.size(A::MPIArray) = A.size
Base.size(A::MPIArray, i::Integer) = A.size[i]
Base.eltype(::MPIArray{T, I, N}) where {T, I, N} = T
Base.getindex(A::MPIArray) = getindex(A.data)
Base.setindex(A::MPIArray, I...) = setindex(A.data, I...)
Base.fill!(A::MPIArray, args...)  = fill!(A.data, args...)

"""
    sync!(A::MPIArray)

    Synchronize data in `A` between MPI ranks.

TBW
"""
function sync!(A::MPIArray)

    np = MPI.Comm_size(A.comm)
    rank = A.myrank
    # begin sync
    req_all = MPI.Request[]
    begin
        for (ghostrank, indices) in A.grank2ghostindices
            req = MPI.Irecv!(view(A.ghostdata, indices...), ghostrank, ghostrank*np + rank, A.comm)
            push!(req_all, req)
        end
        for (remoterank, indices) in A.rrank2localindices
            req = MPI.Isend(A.data[indices...], remoterank, rank*np + remoterank, A.comm)
            push!(req_all, req)
        end
    end
    MPI.Waitall!(req_all)

    MPI.Barrier(A.comm)

    nothing

end

# mpiarray(T::DataType, Asize::NTuple{N, Int}; args...)  where {N<:Integer}  = mpiarray(T, Asize...; args...)

"""
    mpiarray(T::DataType, Asize::Vararg{Int, N}; buffersize = 0, comm = MPI.COMM_WORLD, partitation = (1, MPI.Comm_size(comm))) where {N}

    construct a mpi array with size `Asize` and distributed on MPI `comm` with 
TBW
"""
function mpiarray(T::DataType, Asize::NTuple{N, Int}; buffersize = 0, comm = MPI.COMM_WORLD, 
    partitation = Tuple(map(i -> begin (i < length(Asize)) ? 1 : MPI.Comm_size(comm) end, 1:N))) where {N}

    rank = MPI.Comm_rank(comm)
    np   = MPI.Comm_size(comm)

    allindices = rank2idxs(Asize, partitation)
    rank2indices = Dict(zip(0:(np-1), allindices))
    indices = rank2indices[rank]

    rank2ghostindices = Dict{Int, eltype(values(rank2indices))}()
    for (k, v) in rank2indices
        rank2ghostindices[k] =  map((indice, ub) -> expandslice(indice, buffersize, 1:ub), v, Asize)
    end

    ghostindices = map((indice, ub) -> expandslice(indice, buffersize, 1:ub), indices, Asize)
    ghostdata = zeros(T, map(length, ghostindices)...)

    ghostranks = indice2ranks(ghostindices, rank2indices)
    grank2gindices = grank2ghostindices(ghostranks, ghostindices, rank2indices; localrank = rank)

    remoteranks = indice2ranks(indices, rank2ghostindices)
    rrank2indices = remoterank2indices(remoteranks, indices, rank2ghostindices; localrank = rank)

    dataInGhostData = Tuple(map((i, gi) -> i .- (first(gi) - 1), indices, ghostindices))
    data = view(ghostdata, dataInGhostData...)

    A = MPIArray{T, typeof(indices), N}(data, indices, comm, rank, Asize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    sync!(A)

    MPI.Barrier(comm)
    return A

end