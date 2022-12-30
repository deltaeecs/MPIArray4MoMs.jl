
function Base.:*(A::MPIMatrix, x::AbstractVector)



end

function LinearAlgebra.axpy!(a, X::MPIArray, Y::MPIArray)

    X.indices != Y.indices && throw("This function only supports X and Y with same pattern")

    axpy!(a, X.data, Y.data)

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

    re = MPI.Gather(norm(view(x.parent.dataOffset, x.indices...), p), root, x.parent.comm)
    return MPI.Comm_rank(x.parent.comm) == root ? norm(re, p) : nothing

end


function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:MPIVector}

    ((x.comm != y.comm) || (length(x.data) != length(x.data))) && throw("Vector `x` and `y` must have the same Distribution and length.")
    return MPI.Reduce(dot(x.data, y.data), +, root, x.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:MPIVector}

    ((x.parent.comm != y.comm) || (length(x) != length(y.data))) && throw("Vector `x` and `y` must have the same Distribution and length.")
    return MPI.Reduce(dot(view(x.parent.dataOffset, x.indices...), y.data), +, root, y.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:MPIVector, T2<:SubMPIVector}

    ((x.comm != y.parent.comm) || (length(x.data) != length(y))) && throw("Vector `x` and `y` must have the same Distribution and length.")
    return MPI.Reduce(dot(x.data, view(y.parent.dataOffset, y.indices...)), +, root, x.comm)

end

function LinearAlgebra.dot(x::T1, y::T2; root = 0) where{T1<:SubMPIVector, T2<:SubMPIVector}

    ((x.parent.comm != y.parent.comm) || (length(x) != length(y))) && throw("Vector `x` and `y` must have the same Distribution and length.")
    return MPI.Reduce(dot(view(x.parent.dataOffset, x.indices...), view(y.parent.dataOffset, y.indices...)), +, root, x.parent.comm)

end