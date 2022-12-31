
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

    if root == :all
        return MPI.Allreduce(norm(x.data, p), (args...) -> norm([args...], p), x.comm)
    else
        return MPI.Reduce(norm(x.data, p), (args...) -> norm([args...], p), root, x.comm)
    end

end

function LinearAlgebra.norm(x::T, p::Real=2; root = 0) where{T<:SubMPIArray}
    xlocalidcs = map(intersect, x.indices, x.parent.indices)

    if root == :all
        return MPI.Allreduce(norm(x.parent.dataOffset[xlocalidcs...], p), (args...) -> norm([args...], p), x.parent.comm)
    else
        return MPI.Reduce(norm(x.parent.dataOffset[xlocalidcs...], p), (args...) -> norm([args...], p), root, x.parent.comm)
    end
end


function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:MPIVector}

    if root == :all
        return MPI.Allreduce(dot(x.data, y.data), +,  x.comm)
    else
        return MPI.Reduce(dot(x.data, y.data), +, root, x.comm)
    end

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:MPIVector}

    xlocalidcs = map(intersect, x.indices, x.parent.indices)
    if root == :all
        return MPI.Allreduce(dot(view(x.parent.dataOffset, xlocalidcs...), y.data), +, y.comm)
    else
        return MPI.Reduce(dot(view(x.parent.dataOffset, xlocalidcs...), y.data), +, root, y.comm)
    end

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:SubMPIVector}

    ylocalidcs = map(intersect, y.indices, y.parent.indices)
    if root == :all
        return MPI.Allreduce(dot(x.data, view(y.parent.dataOffset, ylocalidcs...)), +, x.comm)
    else
        return MPI.Reduce(dot(x.data, view(y.parent.dataOffset, ylocalidcs...)), +, root, x.comm)
    end

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:SubMPIVector}

    xlocalidcs = map(intersect, x.indices, x.parent.indices)
    ylocalidcs = map(intersect, y.indices, y.parent.indices)
    if root == :all
        return MPI.Allreduce(dot(view(x.parent.dataOffset, xlocalidcs...), view(y.parent.dataOffset, ylocalidcs...)), +, x.parent.comm)
    else
        return MPI.Reduce(dot(view(x.parent.dataOffset, xlocalidcs...), view(y.parent.dataOffset, ylocalidcs...)), +, root, x.parent.comm)
    end

end