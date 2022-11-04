using MPI
using MPIArray4MoMs
using LinearAlgebra
using OffsetArrays
using BenchmarkTools
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

np = MPI.Comm_size(comm)

A = mpiarray(ComplexF64, (10, 10); partitation = (1, np), buffersize = 3)
A = mpiarray(ComplexF64, 10, 10; partitation = (1, np), buffersize = 3)

fill!(A, rank)
sync!(A)
for (gr, gidcs) in A.grank2ghostindices
    @assert all(A.ghostdata[gidcs...] .== gr)
end

setindex!(A, ComplexF64.(A.indices[1] * A.indices[2]'), :, :)

@assert size(A) == (10, 10)
@assert length(A) == 100

x = mpiarray(ComplexF64, 10; partitation = (np, ), buffersize = 3)

fill!(x, 1)

Av = view(A, 1, A.indices[2])

  x ⋅ x
@show Av ⋅ Av
@show Av ⋅ x
@show x ⋅ Av

t1 = @timed for _ in 1:100000
    x ⋅ x
    MPI.Barrier(x.comm)
end

@show t1.time/100000


t2 = @timed for _ in 1:100000
    Av ⋅ Av
    MPI.Barrier(x.comm)
end

@show t2.time/100000


t2 = @timed for _ in 1:10000
    gather(A)
    MPI.Barrier(x.comm)
end

@show t2.time/10000

Alc = gather(A)

@show Alc

MPI.Finalize()
# display(t)