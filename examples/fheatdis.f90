!> @author Julien Bigot
!> @date   August, 2014
!> @brief  Heat distribution code to test FTI Fortran

program heat
   use FTI
#ifndef MPI_USE_HEADER
   use MPI
#else
   include 'mpif.h'
#endif
   real(8), parameter :: PREC = 0.005
   integer, parameter :: ITER_TIMES = 2000
   integer, parameter :: ITER_OUT = 100
   integer, parameter :: WORKTAG = 50
   integer, parameter :: REDUCE = 5
   integer, parameter :: MEM_MB = 32

   integer, target :: rank, nbProcs, iter, row, col, err, FTI_comm_world
   integer, pointer  :: ptriter
   real(8) :: wtime, memSize, localerror, globalerror
   real(8), pointer :: g(:, :), h(:, :)
  
   integer(8), dimension(3) :: counts
   counts = (/ 11, 22, 33 /)
   globalerror = 1

   call MPI_Init(err)
   FTI_comm_world = MPI_COMM_WORLD
   call FTI_Init('config.fti', FTI_comm_world, err) ! modifies FTI_comm_world
   call MPI_Comm_size(FTI_comm_world, nbProcs, err)
   call MPI_Comm_rank(FTI_comm_world, rank, err)

   row = sqrt((MEM_MB*1024.0*512.0*nbProcs)/8)

   col = (row/nbProcs) + 3
   allocate (g(row, col))
   allocate (h(row, col))

   call initData(g, rank)

   memSize = row*col*2*8/(1024*1024)
   if (rank == 0) then
      print '("Local data size is ",I5," x ",I5," = ",F5.0," MB (",I5,").")', &
         row, col, memSize, MEM_MB
      print '("Target precision : ",F9.5)', PREC
   endif

   ptriter => iter
   call FTI_Protect(0, ptriter, err)
   call FTI_Protect(2, g, err)
   call FTI_SetAttribute( 2, 'temperature_field', FTI_ATTRIBUTE_NAME, err)
   call FTI_SetAttribute( 2, counts, FTI_ATTRIBUTE_DIM, err)
   call FTI_Protect(1, h, err)

   wtime = MPI_Wtime()

   do iter = 1, ITER_TIMES

      call FTI_Snapshot(err)

      call doWork(nbProcs, rank, g, h, localerror)

      if ((mod(iter, ITER_OUT) == 0) .and. (rank == 0)) then
         print '("Step : ",I5,", error = ",F9.5)', iter, globalerror
      endif
      if (mod(iter, REDUCE) == 0) then
         call MPI_Allreduce(localerror, globalerror, 1, MPI_REAL8, MPI_MAX, FTI_comm_world, err)
      endif
      if (globalerror < PREC) exit

   enddo

   if (rank == 0) then
      print '("Execution finished in ",F9.0," seconds.")', MPI_Wtime() - wtime
   endif

   deallocate (h)
   deallocate (g)

   call FTI_Finalize(err)
   call MPI_Finalize(err)

contains

   subroutine initData(h, rank)

      real(8), pointer :: h(:, :)
      integer, intent(IN) :: rank

      integer :: i

      h(:, :) = 0

      if (rank == 0) then
         do i = size(h, 1)/10, 9*size(h, 1)/10
            h(i, 1) = 100
         enddo
      endif

   endsubroutine

   subroutine doWork(numprocs, rank, g, h, localerror)

      integer, intent(IN) :: numprocs
      integer, intent(IN) :: rank
      real(8), pointer :: g(:, :)
      real(8), pointer :: h(:, :)
      real(8), intent(OUT) :: localerror

      integer :: i, j, err, req1(2), req2(2)
      integer :: status1(MPI_STATUS_SIZE, 2), status2(MPI_STATUS_SIZE, 2)

      localerror = 0

      h(:, :) = g(:, :)

      if (rank > 0) then
         call MPI_Isend(g(1, 2), size(g, 1), MPI_REAL8, rank - 1, WORKTAG, FTI_comm_world, req1(1), err)
         call MPI_Irecv(h(1, 1), size(h, 1), MPI_REAL8, rank - 1, WORKTAG, FTI_comm_world, req1(2), err)
      endif
      if (rank < numprocs - 1) then
         call MPI_Isend(g(1, ubound(g, 2) - 1), size(g, 1), MPI_REAL8, rank + 1, WORKTAG, FTI_comm_world, req2(1), err)
         call MPI_Irecv(h(1, ubound(h, 2)), size(h, 1), MPI_REAL8, rank + 1, WORKTAG, FTI_comm_world, req2(2), err)
         !call MPI_Isend(g(ubound(g, 1)-1, 1), size(g, 1), MPI_REAL8, rank+1, WORKTAG, FTI_comm_world, req2(1), err)
         !call MPI_Irecv(h(ubound(h, 1)  , 1), size(h, 1), MPI_REAL8, rank+1, WORKTAG, FTI_comm_world, req2(2), err)
      endif
      if (rank > 0) then
         call MPI_Waitall(2, req1, status1, err)
      endif
      if (rank < numprocs - 1) then
         call MPI_Waitall(2, req2, status2, err)
      endif

      do j = lbound(h, 2) + 1, ubound(h, 2) - 1
         do i = lbound(h, 1), ubound(h, 1) - 1
            g(i, j) = 0.25*(h(i - 1, j) + h(i + 1, j) + h(i, j - 1) + h(i, j + 1))
            if (localerror < abs(g(i, j) - h(i, j))) then
               localerror = abs(g(i, j) - h(i, j))
            endif
         enddo
      enddo

      if (rank == (numprocs - 1)) then
         g(ubound(g, 1), :) = g(ubound(g, 1) - 1, :)
      endif

   endsubroutine

endprogram
