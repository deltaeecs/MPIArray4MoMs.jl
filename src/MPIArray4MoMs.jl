module MPIArray4MoMs

using MPI
using Primes

export  MPIArray,
        mpiarray,
        MPIMatrix,
        MPIVector,
        sync!,
        rank2idxs,
        slicedim2mpi,
        expandslice,
        shrinkslice,
        indice2rank,
        indice2ranks,
        grank2ghostindices,
        remoterank2indices


# 
include("mpiarray.jl")
include("indices.jl")

end # module MPIArray4MoM