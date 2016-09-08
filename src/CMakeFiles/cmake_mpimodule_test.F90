
program test_mpimodule
    use mpi
    integer:: ierr
    call mpi_init(ierr)
    ierr = MPI_BOTTOM
    call mpi_finalize(ierr)
endprogram test_mpimodule
