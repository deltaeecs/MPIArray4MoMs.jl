module MPIArray4MoMs

using MPI
using Primes
using OffsetArrays
using LinearAlgebra

export  MPIArray,
        mpiarray,
        MPIMatrix,
        MPIVector,
        sync!,
        gather,
        rank2idxs,
        slicedim2mpi,
        sizechunks2cuts,
        expandslice,
        shrinkslice,
        indice2rank,
        indice2ranks,
        grank2ghostindices,
        remoterank2indices
        


# Array
include("mpiarray.jl")
# things about index
include("indices.jl")
# LinearAlgebra functions
include("linearalgebra.jl")

end # module MPIArray4MoM