
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
    dataOffset::OffsetArray
    comm::MPI.Comm
    myrank::Int
    size::NTuple{N,Int}
    rank2indices::Dict{Int, I}
    ghostdata::Array{T, N}
    ghostindices::I
    grank2ghostindices::Dict{Int, I}
    rrank2localindices::Dict{Int, I}
end

const MPIVector{T} = MPIArray{T, I, 1} where {T, I}
const MPIMatrix{T} = MPIArray{T, I, 2} where {T, I}
const SubMPIVector{T, SI, L}  =   SubArray{T, 1, MPIArray{T, I, MN}, SI, L} where {T, I, MN, SI, L}
const SubMPIMatrix{T, SI, L}  =   SubArray{T, 2, MPIArray{T, I, MN}, SI, L} where {T, I, MN, SI, L}
const SubMPIArray{T, N, SI, L}  = SubArray{T, N, MPIArray{T, I, MN}, SI, L} where {T, I, N, MN, SI, L}
const SubOrMPIVector{T}  = Union{MPIVector{T}, SubMPIVector{T, SI, L}} where {T, SI, L}
const SubOrMPIMatrix{T}  = Union{MPIMatrix{T}, SubMPIMatrix{T, SI, L}} where {T, SI, L}
const SubOrMPIArray{T, N}= Union{MPIArray{T, I, N}, SubMPIArray{T, I, N, SI, L}} where {T, I, N, SI, L}

Base.size(A::MPIArray) = A.size
Base.size(A::MPIArray, i::Integer) = A.size[i]
Base.length(A::MPIArray) = prod(A.size)
Base.eltype(::MPIArray{T, I, N}) where {T, I, N} = T
Base.getindex(A::MPIArray, I...) = getindex(A.dataOffset, I...)
Base.setindex!(A::MPIArray, X, I...) = setindex!(A.dataOffset, X, I...)
Base.fill!(A::MPIArray, args...)  = fill!(A.data, args...)


mpiarray(T::DataType, Asize::NTuple{N, Int}; args...)  where {N}  = mpiarray(T, Asize...; args...)

"""
    mpiarray(T::DataType, Asize::Vararg{Int, N}; buffersize = 0, comm = MPI.COMM_WORLD, partitation = (1, MPI.Comm_size(comm))) where {N}

    construct a mpi array with size `Asize` and distributed on MPI `comm` with 
TBW
"""
function mpiarray(T::DataType, Asize::Vararg{Int, N}; buffersize = 0, comm = MPI.COMM_WORLD, 
    partitation = Tuple(map(i -> begin (i < length(Asize)) ? 1 : MPI.Comm_size(comm) end, 1:N))) where {N}

    rank = MPI.Comm_rank(comm)
    np   = MPI.Comm_size(comm)

    allindices   = sizeChunks2idxs(Asize, partitation)
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
    data = buffersize == 0 ? ghostdata : view(ghostdata, dataInGhostData...)

    A = MPIArray{T, typeof(indices), N}(data, indices, OffsetArray(data, indices), comm, rank, Asize, rank2indices, ghostdata, ghostindices, grank2gindices, rrank2indices)

    sync!(A)

    MPI.Barrier(comm)
    return A

end

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
    MPI.Waitall(MPI.RequestSet(req_all), MPI.Status)

    MPI.Barrier(A.comm)

    nothing

end


function gather(A::MPIArray; root = 0)

    rank = MPI.Comm_rank(A.comm)

    Alc = rank == root ? zeros(eltype(A), A.size) : nothing

    reqs = [MPI.Isend(Array(A.data), A.comm; dest = root)]
    
    if rank == root
        append!(reqs, map((rk, indices) -> MPI.Irecv!(view(Alc, indices...), A.comm; source = rk), keys(A.rank2indices), values(A.rank2indices)))
    else
        nothing
    end

    MPI.Waitall(MPI.RequestSet(reqs), MPI.Status)

    return Alc

end