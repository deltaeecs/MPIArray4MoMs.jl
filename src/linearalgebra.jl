
function Base.:*(A::MPIMatrix, x::AbstractVector)



end

function LinearAlgebra.axpy!(a, X::MPIArray, Y::MPIArray)

    X.indices != Y.indices && throw("This function only supports X and Y with same pattern")

    axpy!(a, X.data, Y.data)

end

function LinearAlgebra.axpy!(a, X::SubMPIArray, Y::MPIArray)

    xlocalidcs = map(intersect, X.indices, X.parent.indices)
    axpy!(a, view(X.parent.dataOffset, xlocalidcs...), Y.data)

end

function LinearAlgebra.axpy!(a, X::MPIArray, Y::SubMPIArray)

    ylocalidcs = map(intersect, Y.indices, Y.parent.indices)
    axpy!(a, X.data, view(Y.parent.dataOffset, ylocalidcs...))

end

function LinearAlgebra.axpy!(a, X::SubMPIArray, Y::SubMPIArray)

    xlocalidcs = map(intersect, X.indices, X.parent.indices)
    ylocalidcs = map(intersect, Y.indices, Y.parent.indices)
    axpy!(a, view(X.parent.dataOffset, xlocalidcs...), view(Y.parent.dataOffset, ylocalidcs...))

end

function LinearAlgebra.mul!(y, A::T, x::AbstractVector) where{T<:MPIMatrix}
    
    mul!(y, A.data, x)

    return y
end

function LinearAlgebra.norm(x::T, p::Real=2; root = 0) where{T<:MPIArray}

    re = MPI.Gather(norm(x.data, p), root, x.comm)
    return MPI.Comm_rank(x.comm) == root ? norm(re, p) : nothing

end

function LinearAlgebra.norm(x::T, p::Real=2; root = 0) where{T<:SubMPIArray}
    xlocalidcs = map(intersect, X.indices, X.parent.indices)
    re = MPI.Gather(norm(view(x.parent.dataOffset, xlocalidcs...), p), root, x.parent.comm)
    return MPI.Comm_rank(x.parent.comm) == root ? norm(re, p) : nothing

end


function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:MPIVector}

    return MPI.Reduce(dot(x.data, y.data), +, root, x.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:MPIVector}

    xlocalidcs = map(intersect, x.indices, x.parent.indices)
    return MPI.Reduce(dot(view(x.parent.dataOffset, xlocalidcs...), y.data), +, root, y.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:SubMPIVector}

    ylocalidcs = map(intersect, y.indices, y.parent.indices)
    return MPI.Reduce(dot(x.data, view(y.parent.dataOffset, ylocalidcs...)), +, root, x.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:SubMPIVector}

    xlocalidcs = map(intersect, x.indices, x.parent.indices)
    ylocalidcs = map(intersect, y.indices, y.parent.indices)
    return MPI.Reduce(dot(view(x.parent.dataOffset, xlocalidcs...), view(y.parent.dataOffset, ylocalidcs...)), +, root, x.parent.comm)

end