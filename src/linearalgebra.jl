
function Base.:*(A::MPIMatrix, x::AbstractVector)
    throw("`MPIMatrix` 与 `AbstractVector` 的乘法还未实现。")
end

function LinearAlgebra.axpy!(a, X::TX, Y::TY) where{TX<:SubOrMPIArray, TY<:SubOrMPIArray}
    axpy!(a, getdata(X), getdata(Y))
end

function LinearAlgebra.mul!(y, A::T, x::AbstractVector) where{T<:SubOrMPIArray}
    mul!(y, getdata(A), x)
    y
end

function LinearAlgebra.mul!(y::Ty, A::TA, x::AbstractVector) where{Ty<:SubOrMPIArray, TA<:SubOrMPIArray}
    mul!(getdata(y), getdata(A), x)
    y
end

function LinearAlgebra.mul!(y, A::T, x::AbstractVector, α::Number, β::Number) where{T<:SubOrMPIArray}
    mul!(y, getdata(A), x, α, β)
    y
end

function LinearAlgebra.mul!(y::Ty, A::TA, x::AbstractVector, α::Number, β::Number) where{Ty<:SubOrMPIArray, TA<:SubOrMPIArray}
    mul!(getdata(y), getdata(A), x, α, β)
    y
end


function LinearAlgebra.norm(x::T, p::Real=2; root = -1) where{T<:SubOrMPIArray}

    if root == -1
        return MPI.Allreduce(norm(getdata(x), p), (args...) -> norm([args...], p), getcomm(x))
    else
        return MPI.Reduce(norm(getdata(x), p), (args...) -> norm([args...], p), root, getcomm(x))
    end

end

function LinearAlgebra.dot(x::T1, y::T2; root = -1) where{T1<:SubOrMPIArray, T2<:SubOrMPIArray}

    if root == -1
        return MPI.Allreduce(dot(getdata(x), getdata(y)), +,  getcomm(x))
    else
        return MPI.Reduce(dot(getdata(x), getdata(y)), +, root, getcomm(x))
    end

end

function LinearAlgebra.rmul!(A::T1, r::Number) where{T1<:SubOrMPIArray}
    rmul!(getdata(A), r)
    A
end
