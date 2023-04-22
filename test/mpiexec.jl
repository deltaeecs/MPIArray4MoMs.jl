using MPI
using Test
function run_mpi_driver(;procs,file)
  mpidir = @__DIR__
  testdir = mpidir
  repodir = joinpath(testdir,"..")
  nnode = 8
  mpiexec() do cmd
    if MPI.MPI_LIBRARY == "OpenMPI" || (isdefined(MPI, :OpenMPI) && MPI.MPI_LIBRARY == MPI.OpenMPI)
      run(`$cmd --mpi=pmi2 -n $procs --partition=normal --oversubscribe $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    else
      run(`$cmd --mpi=pmi2 -n $procs --nodes=$nnode --ntasks-per-node=$(ceil(Int, procs/nnode))
                 --partition=normal $(Base.julia_cmd()) --project=$repodir $(joinpath(mpidir,file))`)
    end
    # This line will be reached if and only if the command launched by `run` runs without errors.
    # Then, if we arrive here, the test has succeeded.
    @test true  
  end
end
