!> Copyright (c) 2017 Leonardo A. Bautista-Gomez
!> All rights reserved
!>
!> @author Alexandre de Limas Santana (alexandre.delimassantana@bsc.es)
!> @date   August, 2020
!> @brief  Checkpoint primitive types and validate FTI meta-data

program checkpoint_primitives
   use FTI
#ifndef MPI_USE_HEADER
   use MPI
#else
   include 'mpif.h'
#endif

   ! Application parameters
   integer(1), parameter :: SIZE = 5
   character*255         :: configfile
   integer, target       :: err, FTI_comm_world, doInitData, rank

   ! These are the variables for testing
   ! The datatypes should map directly to C primitive types
   ! This should be verified in the meta-data files
   character(1), pointer :: c1(:)

   logical(1), pointer   :: l1(:)
   logical(2), pointer   :: l2(:)
   logical(4), pointer   :: l4(:)
   logical(8), pointer   :: l8(:)

   integer(1), pointer   :: i1(:)
   integer(2), pointer   :: i2(:)
   integer(4), pointer   :: i4(:)
   integer(8), pointer   :: i8(:)

   real(4), pointer      :: r4(:)
   real(8), pointer      :: r8(:)
   real(16), pointer     :: r16(:)

   complex(4), pointer   :: cp4(:)
   complex(8), pointer   :: cp8(:)
   complex(16), pointer  :: cp16(:)

   ! Parse program arguments before starting
   if (command_argument_count() .NE. 2) THEN
      write (*, *) 'Expected two arguments: configfile (str), doInitData (int)'
      stop
   endif

   ! Re-use the 'configfile' variable as a char buffer
   call get_command_argument(2, configfile)
   read (configfile, *) doInitData
   call get_command_argument(1, configfile)

   allocate (c1(SIZE))

   allocate (l1(SIZE))
   allocate (l2(SIZE))
   allocate (l4(SIZE))
   allocate (l8(SIZE))

   allocate (i1(SIZE))
   allocate (i2(SIZE))
   allocate (i4(SIZE))
   allocate (i8(SIZE))

   allocate (r4(SIZE))
   allocate (r8(SIZE))
   allocate (r16(SIZE))

   allocate (cp4(SIZE))
   allocate (cp8(SIZE))
   allocate (cp16(SIZE))

   call MPI_Init(err)
   FTI_comm_world = MPI_COMM_WORLD
   call FTI_Init(configfile, FTI_comm_world, err)
   call MPI_Comm_rank(FTI_comm_world, rank, err)

   ! Conditionally initialize the data
   ! Any initialization pattern will do as long as
   ! the verification logic mimics the patterns
   if (doInitData /= 0) then
      if (rank == 0) then
         print *, "Application initializes its own data"
      end if

      do i = 1, SIZE
         c1(i) = 'A'

         l1(i) = .true.
         l2(i) = .true.
         l4(i) = .true.
         l8(i) = .true.

         i1(i) = 127 - (i-1)
         i2(i) = 32767 - (i-1)
         i4(i) = 2147483647 - (i-1)
         i8(i) = 2147483647 + (i-1)

         r4(i) = 10 + (i-1)
         r8(i) = 20 + (i-1)
         r16(i) = 30 + (i-1)

         cp4(i) = CMPLX(i-1, i-1)
         cp8(i) = CMPLX(i-1, i-1)
         cp16(i) = CMPLX(i-1, i-1)
      end do
   else if (rank == 0) then
      print *, "Application initializes its data with using checkpoints"
   end if

   call FTI_Protect(0, c1, err)

   call FTI_Protect(10, l1, err)
   call FTI_Protect(11, l2, err)
   call FTI_Protect(12, l4, err)
   call FTI_Protect(13, l8, err)

   call FTI_Protect(20, i1, err)
   call FTI_Protect(21, i2, err)
   call FTI_Protect(22, i4, err)
   call FTI_Protect(23, i8, err)

   call FTI_Protect(30, r4, err)
   call FTI_Protect(31, r8, err)
   call FTI_Protect(32, r16, err)

   call FTI_Protect(40, cp4, err)
   call FTI_Protect(41, cp8, err)
   call FTI_Protect(42, cp16, err)

   ! If the application initialized the data:
   ! 1) Checkpoint the data
   ! 2) Simulate a crash
   if (doInitData /= 0) then
      call FTI_Checkpoint(1, 1, err)
      call MPI_Finalize(err)
      call exit(0)
   else
      call FTI_Recover(err)
      if (err /= 0) then
         call exit(1)
      end if
   end if

   ! Test if data is initialized as it should
   err = 0
   do i = 1, SIZE
      if (c1(i) /= 'A') then
         if (rank == 0) then
            print *, "character(1) was corrupted: ", c1(i)
         end if
         call exit(1)
      end if

      if (l1(i) .neqv. .true.) then
         if (rank == 0) then
            print *, "logical(1) was corrupted: ", l1(i)
         end if
         call exit(1)
      end if
      if (l2(i) .neqv. .true.) then
         if (rank == 0) then
            print *, "logical(2) was corrupted: ", l2(i)
         end if
         call exit(1)
      end if
      if (l4(i) .neqv. .true.) then
         if (rank == 0) then
            print *, "logical(4) was corrupted: ", l4(i)
         end if
         call exit(1)
      end if
      if (l8(i) .neqv. .true.) then
         if (rank == 0) then
            print *, "logical(8) was corrupted: ", l8(i)
         end if
         call exit(1)
      end if

      if (i1(i) /= 127 - (i - 1)) then
         if (rank == 0) then
            print *, "integer(1) was corrupted: ", i1(i)
         end if
         call exit(1)
      end if
      if (i2(i) /= 32767 - (i - 1)) then
         if (rank == 0) then
            print *, "integer(2) was corrupted: ", i2(i)
         end if
         call exit(1)
      end if
      if (i4(i) /= 2147483647 - (i - 1)) then
         if (rank == 0) then
            print *, "integer(4) was corrupted: ", i4(i)
         end if
         call exit(1)
      end if
      if (i8(i) /= 2147483647 + (i - 1)) then
         if (rank == 0) then
            print *, "integer(8) was corrupted: ", i8(i)
         end if
         call exit(1)
      end if

      if (r4(i) /= 10 + (i - 1)) then
         if (rank == 0) then
            print *, "real(8) was corrupted: ", r4(i)
         end if
         call exit(1)
      end if
      if (r8(i) /= 20 + (i - 1)) then
         if (rank == 0) then
            print *, "real(8) was corrupted: ", r8(i)
         end if
         call exit(1)
      end if
      if (r16(i) /= 30 + (i - 1)) then
         if (rank == 0) then
            print *, "real(16) was corrupted: ", r16(i)
         end if
         call exit(1)
      end if
      if (cp4(i) /= CMPLX(i-1, i-1)) then
         if (rank == 0) then
            print *, "complex(4) was corrupted: ", cp4(i)
         end if
         call exit(1)
      end if
      if (cp8(i) /= CMPLX(i-1, i-1)) then
         if (rank == 0) then
            print *, "complex(8) was corrupted: ", cp8(i)
         end if
         call exit(1)
      end if
      if (cp16(i) /= CMPLX(i-1, i-1)) then
         if (rank == 0) then
            print *, "complex(16) was corrupted: ", cp16(i)
         end if
         call exit(1)
      end if
   end do

   call FTI_Finalize(err)
   call MPI_Finalize(err)
   call exit(err)
endprogram
