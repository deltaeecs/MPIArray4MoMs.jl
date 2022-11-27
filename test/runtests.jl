using MPIArray4MoMs
using LinearAlgebra
using Test

@testset "MPIArray4MoMs.jl" begin
    
    include("mpiexec.jl")
    run_mpi_driver(procs=8, file="MPIArray.jl")

end
