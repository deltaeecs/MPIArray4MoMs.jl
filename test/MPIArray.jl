using MPI
using MPIArray4MoMs
using LinearAlgebra
using OffsetArrays
using BenchmarkTools

using Test

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
root = 0

np = MPI.Comm_size(comm)

@testset "MPIArray basics" begin
    A = mpiarray(ComplexF64, (10, 10); partitation = (1, np), buffersize = 3)
    @test true
    A = mpiarray(ComplexF64, 10, 10; partitation = (1, np), buffersize = 10)
    @test true
    @test size(A) == (10, 10)
    @test length(A) == 10*10

    A = mpiarray(ComplexF64, 10, 10, 10; partitation = (2, 1, np÷2), buffersize = 3)

    fill!(A, rank)
    sync!(A)
    for (gr, gidcs) in A.grank2ghostindices
        @test all(A.ghostdata[gidcs...] .== gr)
    end

    setindex!(A, fill(rank, size(A.data)), :, :, :)
    sync!(A)
    for (gr, gidcs) in A.grank2ghostindices
        @test all(A.ghostdata[gidcs...] .== gr)
    end


    x = mpiarray(ComplexF64, np - 1; partitation = (np, ), buffersize = 3)
    @test true
    x = mpiarray(ComplexF64, np + 1; partitation = (np, ), buffersize = 3)
    @test true

    fill!(x, 0.1)
    xlc = gather(x)
    @test (rank == 0) ? all(xlc .== 0.1) : isnothing(xlc)

end

@testset "LinearAlgebra" begin

    A = mpiarray(ComplexF64, 10, 10; partitation = (1, np), buffersize = 10)
    x = mpiarray(ComplexF64, 10; partitation = (np, ), buffersize = 10)
    fill!(x, 1)
    fill!(A, 1)

    Av = view(A, 1, A.indices[2])

    @testset "dot" begin
        @test  x ⋅ x    == ((rank == root) ? length(x) : nothing)
        @test Av ⋅ Av   == ((rank == root) ? length(x) : nothing)
        @test Av ⋅ x    == ((rank == root) ? length(x) : nothing)
        @test  x ⋅ Av   == ((rank == root) ? length(x) : nothing)

        @test dot( x,  x; root = :all)   == length(x)
        @test dot(Av, Av; root = :all)   == length(x)
        @test dot(Av,  x; root = :all)   == length(x)
        @test dot( x, Av; root = :all)   == length(x)
    end
    @testset "norm" begin
        @test norm( x)  == ((rank == root) ? sqrt(length(x)) : nothing)
        @test norm(Av)  == ((rank == root) ? sqrt(size(A, 2)) : nothing)

        @test norm( x; root = :all)  == sqrt(length(x))
        @test norm(Av; root = :all)  == sqrt(size(A, 2))

        B = mpiarray(ComplexF64, (10, 10); partitation = (2, np ÷ 2), buffersize = 3)
        fill!(B, 1)
        Bv = view(B, 1, B.indices[2])

        @test norm(Bv)  == ((rank == root) ? sqrt(size(B, 2)) : nothing)
        @test norm(Bv; root = :all)  == sqrt(size(B, 2))
    end

    @testset "axpy!" begin
        y   = deepcopy(x)
        vy  = view(y, :)
        vx  = view(x, :)

        fill!(y, 1)
        axpy!(2, x, y)
        @test  all(y.data .== 3)

        fill!(y, 1)
        axpy!(2, x, vy)
        @test  all(y.data .== 3)

        fill!(y, 1)
        axpy!(2, vx, y)
        @test  all(y.data .== 3)

        fill!(y, 1)
        axpy!(2, vx, vy)
        @test  all(y.data .== 3)
    end


    nepoch = 1000
    t1 = @timed for _ in 1:nepoch
        x ⋅ x
        MPI.Barrier(x.comm)
    end

    t1mean = MPI.Reduce(t1.time/nepoch, +, root, comm)

    rank == root && begin
        t = t1mean/np
        @info "MPI Vector Vector dot mean time = $t s."
    end


    t2 = @timed for _ in 1:nepoch
        Av ⋅ Av
        MPI.Barrier(x.comm)
    end

    t2mean = MPI.Reduce(t2.time/nepoch, +, root, comm)
    rank == root && begin
        t = t2mean/np
        @info "MPI SubVector SubVector dot mean time = $t s."
    end

end


@testset "Transfer" begin
    
    @testset "ArrayTransfer" begin
        A = mpiarray(ComplexF64, (10, 10); partitation = (2, np ÷ 2), buffersize = 0)
        reqsIndices = map((indice, ub) -> expandslice(indice, 3, 1:ub), A.rank2indices[rank], A.size)
        transfer = ArrayTransfer((collect(reqsIndices[1]), reqsIndices[2:end]...), A)
        @test true

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
    end

    @testset "PatternTransfer" begin
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
    end

end

MPI.Finalize()
# display(t)