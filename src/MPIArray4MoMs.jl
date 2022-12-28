module MPIArray4MoMs

using MPI
using Primes
using OffsetArrays
using LinearAlgebra

export  MPIArray, mpiarray,
        MPIMatrix, MPIVector,
        SubMPIVector, SubMPIMatrix, SubMPIArray, 
        SubOrMPIVector, SubOrMPIMatrix, SubOrMPIArray,
        ArrayChunk,
        ArrayTransfer,
        sync!, gather,
        slicedim2mpi, sizeChunks2cuts, sizeChunks2idxs, sizeChunksCuts2indices,
        expandslice, shrinkslice,
        indice2rank, indice2ranks,
        grank2ghostindices, remoterank2indices,
        get_rank2ghostindices, 
        grank2gdataSize, grank2indices
        
# Array
include("mpiarray.jl")
# things about index
include("indices.jl")
# LinearAlgebra functions
include("linearalgebra.jl")

# ArrayChunk
include("arraychunk.jl")
# Array Transfer
include("transfer.jl")

end # module MPIArray4MoM