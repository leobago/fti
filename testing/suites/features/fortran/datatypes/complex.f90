!> Copyright (c) 2017 Leonardo A. Bautista-Gomez
!> All rights reserved
!>
!> @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
!> @date   September, 2020
!> @brief  Checkpoint complex types and validate FTI meta-data
!>
!> @details
!> This program tests the use of the following functions in Fortran:
!> - FTI_InitType;
!> - FTI_InitCompositeType;
!> - FTI_AddScalarField;
!> - FTI_AddVectorField;
!> - FTI_Protect (with user-defined data types).
!>
!> We define structures that compose different use cases for these functions.
!> - Foo has only simple types;
!> - Bar encapsulates a Foo structure;
!> - BarVec encapsulates an array of Foo structures.
!>
!> The program consists of array and scalar initializations for these types.
!> In the first execution, the doInitData must be true.
!> The values are initialized and checkpointed in the first execution.
!> In the second execution, doInitData must be false.
!> The values are recovered and checked in the second execution.
!>
!> This program can be a nice tool to validate/visualize FTI/HDF5 integration.
!> To do this, run with doInitData and use h5dump on the checkpoint files.

program checkpoint_complex
   use ISO_C_BINDING
   use FTI
#ifndef MPI_USE_HEADER
   use MPI
#else
   include 'mpif.h'
#endif

   type :: foo
      integer(4) :: A
      real(4)    :: B
   end type

   type :: bar
      type(foo)               :: foos
      integer(2)              :: x
   end type

   type :: barvec
      type(foo), dimension(2) :: foos
      integer(2)              :: x
   end type

   ! Application Parameters
   character*255           :: configfile
   integer                 :: doInitData, corrupted
   integer, parameter      :: SIZE = 3
   ! FTI data
   integer                 :: foo_tid, foo_complex_tid, bar_tid, barvec_tid
   integer, target         :: ierr, rank, FTI_COMM_WORLD
   ! Application data
   type(foo), target                       :: single
   type(foo), dimension(SIZE), target      :: vec
   type(foo), dimension(SIZE, SIZE), target :: matrix
   type(foo), target                       :: single_described
   type(bar), target                       :: container
   type(barvec), target                    :: cont_vec
   integer, dimension(1), target           :: list

   ! Parse program arguments before starting
   if (command_argument_count() .NE. 2) THEN
      write (*, *) 'Expected two arguments: configfile (str), doInitData (int)'
      stop
   endif
   call get_command_argument(2, configfile)
   read (configfile, *) doInitData
   call get_command_argument(1, configfile)

   ! Start program
   call MPI_Init(ierr)
   FTI_comm_world = MPI_COMM_WORLD
   call FTI_Init(configfile, FTI_comm_world, ierr)
   call MPI_Comm_rank(FTI_comm_world, rank, ierr)
   corrupted = 0

   ! Declare simple and complex user types
   foo_tid = FTI_InitType(sizeof(single))
   foo_complex_tid = FTI_InitCompositeType("foo", sizeof(single_described))
   bar_tid = FTI_InitCompositeType("bar", sizeof(container))
   barvec_tid = FTI_InitCompositeType("barvec", sizeof(cont_vec))

   ! Populate complex type definitions for FOO
   ierr = FTI_AddScalarField(foo_complex_tid, "A", FTI_INTEGER4, int(0, c_size_t))
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if
   ierr = FTI_AddScalarField(foo_complex_tid, "B", FTI_REAL4, sizeof(single%A))
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if
   ! Populate complex type definitions for BAR
   ierr = FTI_AddScalarField(bar_tid, "foos", foo_complex_tid, int(0, c_size_t))
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if
   ierr = FTI_AddScalarField(bar_tid, "x", FTI_INTEGER2, sizeof(container%foos))
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if
   ! Populate complex type definitions for BARVEC
   list = 2
   ierr = FTI_AddVectorField(barvec_tid, "foos", foo_complex_tid, int(0, c_size_t), 1, list)
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if
   ierr = FTI_AddScalarField(barvec_tid, "x", FTI_INTEGER2, sizeof(cont_vec%foos))
   if (ierr == FTI_NSCS) then
      call exit(1)
   end if

   call FTI_Protect(1, c_loc(single), 1, foo_tid, ierr)
   call FTI_Protect(2, c_loc(vec), SIZE, foo_tid, ierr)
   call FTI_Protect(3, c_loc(matrix), SIZE*SIZE, foo_tid, ierr)
   call FTI_Protect(4, c_loc(single_described), 1, foo_complex_tid, ierr)
   call FTI_Protect(5, c_loc(container), 1, bar_tid, ierr)
   call FTI_Protect(6, c_loc(cont_vec), 1, barvec_tid, ierr)

   call FTI_Status(ierr)
   if (doInitData .eq. 1) then
      single%A = 1
      single%B = 2.0
      single_described%A = 100
      single_described%B = 200.0
      container%foos%A = 1000
      container%foos%B = 1000 + 0.1
      container%x = 99
      cont_vec%foos(1)%A = 42
      cont_vec%foos(1)%B = 24
      cont_vec%foos(2)%A = 37
      cont_vec%foos(2)%B = 73
      cont_vec%x = 101
      do i = 1, SIZE
         vec(i)%A = i
         vec(i)%B = i + 0.1
         do j = 1, SIZE
            matrix(i, j)%A = i + 10*j
            matrix(i, j)%B = i + 10*j + 0.1
         end do
      end do
      call FTI_Checkpoint(1, 1, ierr)
      call MPI_Barrier(FTI_COMM_WORLD, ierr)
      if (ierr == FTI_NSCS .and. rank == 0) then
         print *, "Failed to create checkpoint"
      end if
   else
      call FTI_Recover(ierr)
      ! Check variable: single
      if (rank == 0) then
         if (single%A /= 1 .or. single%B /= 2.0) then
            print *, "single was corrupted."
         end if
         print *, "single: ", single%A, ", ", single%B

         ! Check variable: single_described
         if (single_described%A /= 100 .or. single_described%B /= 200.0) then
            print *, "single described was corrupted."
            corrupted = 1
         end if
         print *, "single_described: ", single_described%A, ", ", single_described%B
         ! Check variable: container
         if (container%foos%A /= 1000 .or. container%foos%B /= 1000 + 0.1 .or. container%x /= 99) then
            print *, "container was corrupted."
            corrupted = 1
         end if
         print *, "container: ", container%foos%A, ", ", container%foos%B, ", ", container%x
         ! Check variable: cont_vec
         if (cont_vec%foos(1)%A /= 42 .or. cont_vec%foos(1)%B /= 24 .or. &
             cont_vec%foos(2)%A /= 37 .or. cont_vec%foos(2)%B /= 73 .or. &
             cont_vec%x /= 101) then
            print *, "foo vector container was corrupted."
            corrupted = 1
         end if
         print *, "foo vector container: "
         print *, "First: ", cont_vec%foos(1)%A, ", ", cont_vec%foos(1)%B
         print *, "Second: ", cont_vec%foos(2)%A, ", ", cont_vec%foos(2)%B
         print *, "X: ", cont_vec%x
         ! Check variable: vec
         do i = 1, SIZE
            if (vec(i)%A /= i .or. vec(i)%B /= i + 0.1) then
               print *, "vector was corrupted."
               corrupted = 1
            end if
            print *, "vector(", i, "): ", vec(i)%A, ", ", vec(i)%B
            ! Check variable matrix
            do j = 1, SIZE
               if (matrix(i, j)%A /= i + 10*j .or. matrix(i, j)%B /= i + 10*j + 0.1) then
                  print *, "matrix was corrupted."
                  corrupted = 1
               end if
               print *, "matrix(", i, ",", j, "): ", matrix(i, j)%A, ", ", matrix(i, j)%B
            end do
         end do
      end if
      call FTI_Finalize(ierr)
   end if
   call MPI_Finalize(ierr)
   call exit(corrupted)
end program
