"""
    创建特殊用途的 data Transfer, 用于层内不同 rank 的Pattern 传输数据
    Independt data transfer for MPIArray.
    parent::MPIArray the parent array to transfer
    reqsIndices::NTuple{N, Union{UnitRange{Int}, Vector{Int}}}
    reqsDatas::Dict{Int, SparseMatrixCSC{T, Int}} 用 SparseMatrixCSC 保存 pole 和 θϕ 方向的数据，Dict建保存 cube 数据。
    recv_rk2idcs::Dict{Int, I}
    send_rk2idcs::Dict{Int, I}

"""
struct PatternTransfer{T, I} <: TRANSFER
    parent::MPIArray{T, IA, 3} where {IA}
    reqsIndices::NTuple{3, Union{UnitRange{Int}, Vector{Int}}}
    reqsDatas::Dict{Int, SparseMatrixCSC{T, Int}}
    recv_rk2idcs::Dict{Int, I}
    send_rk2idcs::Dict{Int, I}
end


"""
    PatternTransfer(reqsIndices::NTuple{N, I}, a::MPIArray{T, IA, N}; comm = a.comm, rank = a.myrank, np = MPI.Comm_size(comm)) where {N, I, T, IA}

    创建缓冲区存储 Pattern 交换数据.
TBW
"""
function PatternTransfer(reqsIndices::NTuple{3, Union{UnitRange{Int}, Vector{Int}}}, a::MPIArray{T, IA, 3}; comm = a.comm, rank = a.myrank, np = MPI.Comm_size(comm)) where {T, IA}
    
    # 收集所有进程需求的 indices
    all_reqIndices      =   map(idc -> allgather_VecOrUnitRange(idc; comm = comm), reqsIndices)
    rank2reqIndices     =   Dict(zip(0:(np-1), zip(all_reqIndices...)))
    rank2indices        =   a.rank2indices

    # 需要接收的 rank 和数据的 indices.
    recv_ranks  =   indice2ranks(reqsIndices, rank2indices)
    recv_rank2indices   =   grank2indices(recv_ranks, reqsIndices, rank2indices)

    ## 先创建稀疏矩阵的行列索引并保存在字典里
    # 先找出所有的 cube id
    cubeIndices = Int[]
    for rkcubeIdc in values(recv_rank2indices)
        unique!(sort!(append!(cubeIndices, rkcubeIdc[3])))
    end
    
    cubeIdc2IsJs = Dict([i => (Is = Int[], Js = Int[]) for i in cubeIndices])

    # 其次找出所有 cube 的索引
    for idcs in values(recv_rank2indices)
        for iCube in idcs[3]
            Is, Js = cubeIdc2IsJs[iCube]
            append!(Is, repeat(idcs[1], outer = length(idcs[2])))
            append!(Js, repeat(idcs[2], inner = length(idcs[1])))
        end
    end

    reqsDatas = Dict([i => sparse(  cubeIdc2IsJs[i]..., zeros(T, length(cubeIdc2IsJs[i].Is)), 
                                    size(a, 1), size(a, 2)) for i in cubeIndices])
    
    # 需要发送的 rank 和数据在 a.data 内的 indice
    send_ranks = indice2ranks(a.indices, rank2reqIndices)
    send_rank2indices = remoterank2indices(send_ranks, a.indices, rank2reqIndices)

    PatternTransfer{T, eltype(values(recv_rank2indices))}(a, reqsIndices, reqsDatas, recv_rank2indices, send_rank2indices)

end

"""
    sync!(t::PatternTransfer; comm = t.parent.comm, rank = t.parent.myrank, np = MPI.Comm_size(comm))

    sync data in t.
TBW
"""
function sync!(t::PatternTransfer{T, I}; comm = t.parent.comm, rank = t.parent.myrank, np = MPI.Comm_size(comm)) where{T, I}

    # parent mpi array
    A = t.parent
    # restoring region
    reqsDatas = t.reqsDatas
    # 缓冲数据，用于接收数据然后再赋值给 reqsDatas 的稀疏矩阵
    buffers =  Dict(k => zeros(T, map(length, v)) for (k, v) in t.recv_rk2idcs)

    # begin sync
    req_all = MPI.Request[]
    begin
        for ghostrank in keys(t.recv_rk2idcs)
            req = MPI.Irecv!(buffers[ghostrank], ghostrank, ghostrank*np + rank, A.comm)
            push!(req_all, req)
        end
        for (remoterank, indices) in t.send_rk2idcs
            req = MPI.Isend(A.data[indices...], remoterank, rank*np + remoterank, A.comm)
            push!(req_all, req)
        end
    end
    MPI.Waitall(MPI.RequestSet(req_all), MPI.Status)
    
    # 将数据存储进
    for (ghostrank, indices) in t.recv_rk2idcs
        buffer = buffers[ghostrank]
        for (i, iCube) in enumerate(indices[3])
            setindex!(reqsDatas[iCube], view(buffer, :, :, i), indices[1], indices[2])
        end
    end
    MPI.Barrier(A.comm)

    return nothing

end