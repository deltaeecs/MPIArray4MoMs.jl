using MPI
using MPIArray4MoMs
using BenchmarkTools
using Statistics
# using OrderedCollections
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

np = MPI.Comm_size(comm)

A = mpiarray(ComplexF64, (10, 10); partitation = (1, np), buffersize = 1)
fill!(A, rank)
sync!(A)

for (gr, gidcs) in A.grank2ghostindices
    @assert all(A.ghostdata[gidcs...] .== gr)
end

MPI.Finalize()
# display(t)