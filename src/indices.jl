function expandslice(idxs, bandwidth, bounds)

    d = max(first(bounds), first(idxs) - bandwidth)
    u = min(last(bounds), last(idxs) + bandwidth)

    d:u

end


function slicedim2mpi(dims, pids)
    dims = [dims...]
    chunks = ones(Int, length(dims))
    np = length(pids)
    f = sort!(collect(keys(factor(np))), rev=true)
    k = 1
    while np > 1
        # repeatedly allocate largest factor to largest dim
        if np % f[k] != 0
            k += 1
            if k > length(f)
                break
            end
        end
        fac = f[k]
        (d, dno) = findmax(dims)
        # resolve ties to highest dim
        dno = findlast(isequal(d), dims)
        if dims[dno] >= fac
            dims[dno] = div(dims[dno], fac)
            chunks[dno] *= fac
        end
        np = div(np, fac)
    end
    return chunks
end


function slicedim2mpi(sz::Int, nc::Int)
    if sz >= nc
        chunk_size = div(sz,nc)
        remainder = rem(sz,nc)
        grid = zeros(Int64, nc+1)
        for i = 1:(nc+1)
            grid[i] += (i-1)*chunk_size + 1
            if i<= remainder
                grid[i] += i-1
            else
                grid[i] += remainder
            end
        end
        return grid
    else
        return [[1:(sz+1);]; zeros(Int, nc-sz)]
    end
end

function sizeChunks2cuts(Asize, chunks::Tuple)
    map(slicedim2mpi, Asize, chunks)
end

function sizeChunks2cuts(Asize, chunks::Int)
    map(slicedim2mpi, Asize, (chunks, ))
end

function sizeChunksCuts2indices(Asize, chunks, cuts::Tuple)
    n = length(Asize)
    idxs = Array{NTuple{n,UnitRange{Int}}, n}(undef, chunks...)
    for cidx in CartesianIndices(tuple(chunks...))
        if n > 0
            idxs[cidx.I...] = ntuple(i -> (cuts[i][cidx[i]]:cuts[i][cidx[i] + 1] - 1), n)
        else
            throw("0 dim array not supported.")
        end
    end

    return idxs
end

function sizeChunksCuts2indices(Asize, chunks, cuts::Vector{I}) where{I<:Integer}
    n = length(Asize)
    idxs = Array{NTuple{n,UnitRange{Int}}, n}(undef, chunks...)
    for cidx in CartesianIndices(tuple(chunks...))
        idxs[cidx.I...] = (cuts[cidx[1]]:cuts[cidx[1] + 1] - 1, )
    end

    return idxs
end


"""
    sizeChunks2idxs(Asize, chunks)

    Borrowed form DistributedArray.jl, get the slice of matrix
    size Asize on each dimension with chunks.

TBW
"""
function sizeChunks2idxs(Asize, chunks)
    cuts = sizeChunks2cuts(Asize, chunks)
    return sizeChunksCuts2indices(Asize, chunks, cuts)
end


"""
    indice2rank(indice::T, rank2indices::Dict{Integer, NTuple{1}}) where{T<:Integer}

    Get the rank of indice::Int form rank2indices
TBW
"""
function indice2rank(indice::T, rank2indices::Dict{Integer, NTuple{1}}) where{T<:Integer}

    rks  = findall(x -> indice in x, rank2indices)
    
    re = intersect(rks...)

    isempty(re) && throw("No suitable rank found, please recheck!")
    length(re) > 1  && throw("Multi ranks found, please recheck!")

    return re[1]

end

"""
    indice2rank(indice::NTuple{N, T}, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}) where{T<:Integer, N, T2}
    
    Get the rank of indice::Ntuple{N, Int} form rank2indices
TBW
"""
function indice2rank(indice::NTuple{N, T}, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}) where{T<:Integer, N, T2}

    rks  = map(i -> findall(x -> indice[i] in x[i], rank2indices), 1:N)
    
    re = intersect(rks...)

    # isempty(re) && throw("No suitable rank found, please recheck!")
    length(re) > 1  && throw("Multi ranks found, please recheck!")

    return re[1]

end


"""
    indice2ranks(indice::NTuple{N, T}, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}) where{T<:Integer, N, T2}
        
    Get the rank of indice::Ntuple{N, Indice} form rank2indices
TBW
"""
function indice2ranks(indice::NTuple{N, T}, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}) where{T, N, T2}

    rks  = map(i -> findall(x -> !isempty(intersect(indice[i], x[i])), rank2indices), 1:N)
    
    re = intersect(rks...)
    # isempty(re) && throw("No suitable rank found, please recheck!")

    return re

end


"""
    grank2ghostindices(ghostranks, ghostindices, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}; localrank = MPI.Comm_rank(MPI.COMM_WORLD)) where{N, T2}

    get indices of ghost data in its hosting rank.

TBW
"""
function grank2ghostindices(ghostranks, ghostindices::Tuple{Vararg{T1, N}}, rank2indices::Dict{Int, Tuple{Vararg{T2, N}}}; localrank = MPI.Comm_rank(MPI.COMM_WORLD)) where{N, T1, T2}

    grank2gindices = Dict{Int, Tuple{Vararg{T1, N}}}()
    for grank in ghostranks
        grank == localrank && continue
        grank2gindices[grank] = Tuple([intersect(rank2indices[grank][i], ghostindices[i]) .- (first(ghostindices[i]) - 1) for i in 1:N])
    end

    return grank2gindices
end

"""
    remoterank2indices(rank, remoteranks, rank2ghostindices::Dict{Int, Tuple{Vararg{T2, N}}}; localrank = MPI.Comm_rank(MPI.COMM_WORLD)) where{N, T2}

    get remote rank and relative indices in data.

TBW
"""
function remoterank2indices(remoteranks, indices, rank2ghostindices::Dict{Int, Tuple{Vararg{T2, N}}}; localrank = MPI.Comm_rank(MPI.COMM_WORLD)) where{N, T2}

    rrank2indices = Dict{Int, Tuple{Vararg{T2, N}}}()
    for rrank in remoteranks
        rrank == localrank && continue
        rrank2indices[rrank] = Tuple([intersect(rank2ghostindices[rrank][i], indices[i]) .- (first(indices[i]) - 1) for i in 1:N])
    end

    return rrank2indices
end