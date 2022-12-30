using MPI
using MPIArray4MoMs
using LinearAlgebra
using OffsetArrays
using BenchmarkTools

using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

np = MPI.Comm_size(comm)

A = mpiarray(ComplexF64, (10, 10); partitation = (1, np), buffersize = 3)
A = mpiarray(ComplexF64, 10, 10; partitation = (1, np), buffersize = 10)

fill!(A, rank)
sync!(A)
for (gr, gidcs) in A.grank2ghostindices
    @test all(A.ghostdata[gidcs...] .== gr)
end

setindex!(A, ComplexF64.(A.indices[1] * A.indices[2]'), :, :)

@test size(A) == (10, 10)
@test length(A) == 10*10


x = mpiarray(ComplexF64, np - 1; partitation = (np, ), buffersize = 3)
x = mpiarray(ComplexF64, np + 1; partitation = (np, ), buffersize = 3)
x = mpiarray(ComplexF64, 10; partitation = (np, ), buffersize = 10)
fill!(x, 1)
fill!(A, 1)

Av = view(A, 1, A.indices[2])

@test  x ⋅ x    == ((rank == 0) ? length(x) : nothing)
@test Av ⋅ Av   == ((rank == 0) ? length(x) : nothing)
@test Av ⋅ x    == ((rank == 0) ? length(x) : nothing)
@test  x ⋅ Av   == ((rank == 0) ? length(x) : nothing)

t1 = @timed for _ in 1:1000
    x ⋅ x
    MPI.Barrier(x.comm)
end

@info rank t1.time/1000


t2 = @timed for _ in 1:1000
    Av ⋅ Av
    MPI.Barrier(x.comm)
end

@info rank t2.time/1000

xlc = gather(x)
@test (rank == 0) ? all(xlc .== 1) : isnothing(xlc)


## ArrayTransfer
A = mpiarray(ComplexF64, (10, 10); partitation = (2, np ÷ 2), buffersize = 0)
reqsIndices = map((indice, ub) -> expandslice(indice, 3, 1:ub), A.rank2indices[rank], A.size)
transfer = ArrayTransfer((collect(reqsIndices[1]), reqsIndices[2:end]...), A)

fill!(A, rank)
sync!(transfer)

for rk in 1:np
    rk == (rank + 1) && begin
        for (rk, a) in transfer.reqsDatas
            @test all(a.data .== rk)
        end
    end
    MPI.Barrier(comm)
end

## PatternTransfer
A = mpiarray(ComplexF64, (10, 10, 10); partitation = (2, 2, np ÷ 4), buffersize = 0)
reqsIndices = map((indice, ub) -> expandslice(indice, 3, 1:ub), A.rank2indices[rank], A.size)

transfer = PatternTransfer((collect(reqsIndices[1]), reqsIndices[2:end]...), A)

fill!(A, rank)
MPI.Barrier(comm)
sync!(transfer)

reqsDatas = transfer.reqsDatas
for rk in 1:np
    rk == (rank + 1) && begin
        for (rk, idcs) in transfer.recv_rk2idcs
            for iCube in idcs[3]
                a = reqsDatas[iCube][idcs[1], idcs[2]]
                @test all(a.nzval .== rk)
            end
        end
    end
    MPI.Barrier(comm)
end

MPI.Finalize()
# display(t)