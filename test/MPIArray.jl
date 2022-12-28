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
A = mpiarray(ComplexF64, 10, 10; partitation = (1, np), buffersize = 10)

fill!(A, rank)
sync!(A)
for (gr, gidcs) in A.grank2ghostindices
    @assert all(A.ghostdata[gidcs...] .== gr)
end

setindex!(A, ComplexF64.(A.indices[1] * A.indices[2]'), :, :)

@assert size(A) == (10, 10)
@assert length(A) == 10*10


x = mpiarray(ComplexF64, np - 1; partitation = (np, ), buffersize = 3)
x = mpiarray(ComplexF64, np + 1; partitation = (np, ), buffersize = 3)
x = mpiarray(ComplexF64, 10; partitation = (np, ), buffersize = 10)
fill!(x, 1)

Av = view(A, 1, A.indices[2])

@show  x ⋅ x
@show Av ⋅ Av
@show Av ⋅ x
@show  x ⋅ Av

t1 = @timed for _ in 1:1000
    x ⋅ x
    MPI.Barrier(x.comm)
end

@show t1.time/1000


t2 = @timed for _ in 1:1000
    Av ⋅ Av
    MPI.Barrier(x.comm)
end

@show t2.time/1000


# t3 = @timed for _ in 1:10
#     gather(A)
#     MPI.Barrier(x.comm)
# end

# @show t3.time/10

# fill!(A, rank+1)

# Alc = gather(A)

# display(Alc)

xlc = gather(x)

display(xlc)


A = mpiarray(ComplexF64, (10, 10); partitation = (2, np ÷ 2), buffersize = 0)
reqsIndices = map((indice, ub) -> expandslice(indice, 3, 1:ub), A.rank2indices[rank], A.size)
transfer = ArrayTransfer((collect(reqsIndices[1]), reqsIndices[2:end]...), A)


fill!(A, rank)
MPI.Barrier(comm)
sync!(transfer)

for rk in 1:np
    rk == (rank + 1) && begin
        @info rank reqsIndices
        for (rk, a) in transfer.reqsDatas
            @info rank rk size(a) sum(a)/length(a)
        end
    end
    MPI.Barrier(comm)
end

MPI.Finalize()
# display(t)